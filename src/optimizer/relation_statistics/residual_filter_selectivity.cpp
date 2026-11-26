#include "duckdb/optimizer/relation_statistics/residual_filter_selectivity.hpp"

#include "duckdb/common/limits.hpp"
#include "duckdb/optimizer/relation_statistics/relation_statistics_helper.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"

namespace duckdb {

struct ResidualFilterEstimator::ExpressionEstimate {
	CardinalityValue selectivity = 1;
	bool has_evidence = false;
	bool has_fallback = false;
	column_binding_set_t evidence_columns;
	optional_idx single_relation_idx;
	CardinalityValue single_relation_cardinality = 0;

	CardinalityValue Effective(CardinalityValue default_selectivity) const {
		return has_fallback ? selectivity * default_selectivity : selectivity;
	}
};

namespace {

bool IsLocalColumnRef(const Expression &expr) {
	return expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
	       expr.Cast<BoundColumnRefExpression>().Depth() == 0;
}

bool TryGetSingleColumnBinding(const Expression &expr, ColumnBinding &out_binding) {
	bool found = false;
	bool valid = true;
	ExpressionIterator::VisitExpression<BoundColumnRefExpression>(expr, [&](const BoundColumnRefExpression &col_ref) {
		if (!valid) {
			return;
		}
		if (col_ref.Depth() != 0 || (found && out_binding != col_ref.Binding())) {
			valid = false;
			return;
		}
		out_binding = col_ref.Binding();
		found = true;
	});
	return valid && found;
}

} // namespace

ResidualFilterEstimator::ResidualFilterEstimator(const vector<RelationStats> &relations,
                                                 const column_binding_map_t<ColumnBinding> &equivalence_roots) {
	for (idx_t i = 0; i < relations.size(); i++) {
		auto &stats = relations[i];
		auto cardinality = stats.GetCardinality();
		if (cardinality.IsZero()) {
			continue;
		}
		for (idx_t col_idx = 0; col_idx < stats.columns.size(); col_idx++) {
			auto &col = stats.columns[col_idx];
			// Equality bounds can shrink the current domain, so reliability comes from the total domain.
			auto source = col.total_domain.source;
			if (source != DistinctCountSource::HLL && source != DistinctCountSource::EXACT) {
				continue;
			}
			auto domain = MinValue(cardinality, col.current_domain.distinct_count);
			if (domain.IsZero()) {
				continue;
			}
			auto normalized = ColumnBinding(TableIndex(i), ProjectionIndex(col_idx));
			auto eq_entry = equivalence_roots.find(normalized);
			auto canonical = eq_entry != equivalence_roots.end() ? eq_entry->second : normalized;
			domains.emplace(col.binding, ColumnDomain {i, cardinality, domain, canonical});
		}
	}
}

ResidualFilterEstimator::ExpressionEstimate ResidualFilterEstimator::EstimateExpression(const Expression &expr,
                                                                                        idx_t depth) const {
	static constexpr idx_t MAX_RESIDUAL_DEPTH = 32;
	if (depth >= MAX_RESIDUAL_DEPTH) {
		return {1, false, true, {}};
	}
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT && expr.GetReturnType() == LogicalType::BOOLEAN) {
		auto &val = expr.Cast<BoundConstantExpression>().GetValue();
		return (val.IsNull() || !val.GetValue<bool>()) ? ExpressionEstimate {0, true, false, {}}
		                                               : ExpressionEstimate {1, false, false, {}};
	}
	if (expr.GetExpressionType() == ExpressionType::CONJUNCTION_AND &&
	    expr.GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
		vector<reference<const Expression>> conjuncts;
		vector<reference<const Expression>> stack {expr};
		while (!stack.empty()) {
			auto &current = stack.back().get();
			stack.pop_back();
			if (current.GetExpressionType() == ExpressionType::CONJUNCTION_AND &&
			    current.GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
				auto &children = current.Cast<BoundConjunctionExpression>().GetChildren();
				for (idx_t i = children.size(); i > 0; i--) {
					stack.push_back(*children[i - 1]);
				}
			} else {
				conjuncts.push_back(current);
			}
		}
		struct ColumnEvidence {
			CardinalityValue selectivity;
			optional_idx relation_idx;
			CardinalityValue relation_cardinality;
		};
		column_binding_map_t<ColumnEvidence> column_selectivities;
		column_binding_set_t other_evidence_columns;
		CardinalityValue other_selectivity(1);
		bool has_evidence = false;
		bool has_fallback = false;
		bool has_overlap = false;
		for (auto &conjunct : conjuncts) {
			auto sub = EstimateExpression(conjunct.get(), depth + 1);
			if (sub.selectivity.IsZero()) {
				return {0, true, false, {}};
			}
			has_fallback |= sub.has_fallback;
			if (!sub.has_evidence) {
				continue;
			}
			has_evidence = true;
			if (!sub.has_fallback && sub.evidence_columns.size() == 1) {
				auto col = *sub.evidence_columns.begin();
				auto res = column_selectivities.emplace(
				    col, ColumnEvidence {sub.selectivity, sub.single_relation_idx, sub.single_relation_cardinality});
				if (!res.second && sub.selectivity < res.first->second.selectivity) {
					res.first->second = {sub.selectivity, sub.single_relation_idx, sub.single_relation_cardinality};
				}
				continue;
			}
			D_ASSERT(!sub.evidence_columns.empty());
			for (auto &col : sub.evidence_columns) {
				if (!other_evidence_columns.insert(col).second) {
					has_overlap = true;
				}
			}
			other_selectivity *= sub.selectivity;
		}
		for (auto &entry : column_selectivities) {
			if (other_evidence_columns.count(entry.first)) {
				has_overlap = true;
				break;
			}
		}
		if (has_overlap) {
			return {1, false, true, {}};
		}
		// Bound same-relation column products by 1 / cardinality to limit correlated underestimation.
		unordered_map<idx_t, pair<CardinalityValue, CardinalityValue>> relation_selectivities;
		auto total = other_selectivity;
		for (auto &entry : column_selectivities) {
			auto &ev = entry.second;
			if (ev.relation_idx.IsValid()) {
				auto rel = relation_selectivities.emplace(ev.relation_idx.GetIndex(),
				                                          make_pair(CardinalityValue(1), ev.relation_cardinality));
				rel.first->second.first *= ev.selectivity;
			} else {
				total *= ev.selectivity;
			}
			other_evidence_columns.insert(entry.first);
		}
		for (auto &entry : relation_selectivities) {
			auto floor = entry.second.second <= 1 ? CardinalityValue(1) : CardinalityValue(1) / entry.second.second;
			total *= MaxValue(entry.second.first, floor);
		}
		optional_idx single_rel;
		CardinalityValue single_card(0);
		if (!has_fallback && other_evidence_columns.size() == 1 && column_selectivities.size() == 1) {
			single_rel = column_selectivities.begin()->second.relation_idx;
			single_card = column_selectivities.begin()->second.relation_cardinality;
		}
		return {total, has_evidence, has_fallback, std::move(other_evidence_columns), single_rel, single_card};
	}
	if (expr.GetExpressionType() == ExpressionType::CONJUNCTION_OR &&
	    expr.GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
		const CardinalityValue default_selectivity(RelationStatisticsHelper::DEFAULT_SELECTIVITY);
		CardinalityValue total_selectivity(0);
		bool has_surviving = false;
		bool has_unsupported = false;
		column_binding_set_t evidence_columns;
		optional_idx shared_rel;
		CardinalityValue shared_card(0);
		bool consistent_rel = true;
		for (auto &child : expr.Cast<BoundConjunctionExpression>().GetChildren()) {
			auto sub = EstimateExpression(*child, depth + 1);
			if (sub.selectivity.IsZero()) {
				continue;
			}
			if (!sub.has_evidence) {
				if (!sub.has_fallback) {
					return {1, false, false, {}};
				}
				has_unsupported = true;
				continue;
			}
			total_selectivity += sub.Effective(default_selectivity);
			evidence_columns.insert(sub.evidence_columns.begin(), sub.evidence_columns.end());
			if (!sub.single_relation_idx.IsValid()) {
				consistent_rel = false;
			} else if (!has_surviving) {
				shared_rel = sub.single_relation_idx;
				shared_card = sub.single_relation_cardinality;
			} else if (shared_rel != sub.single_relation_idx) {
				consistent_rel = false;
			}
			has_surviving = true;
		}
		if (has_unsupported) {
			return {1, false, true, {}};
		}
		if (!has_surviving) {
			return {0, true, false, {}};
		}
		if (total_selectivity >= 1) {
			return {1, false, false, {}};
		}
		return {total_selectivity,
		        true,
		        false,
		        std::move(evidence_columns),
		        consistent_rel ? shared_rel : optional_idx(),
		        consistent_rel ? shared_card : CardinalityValue(0)};
	}
	if (expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NOT_NULL &&
	    expr.GetExpressionClass() == ExpressionClass::BOUND_OPERATOR) {
		auto &children = expr.Cast<BoundOperatorExpression>().GetChildren();
		if (children.size() == 1 && IsLocalColumnRef(*children[0])) {
			return {1, false, false, {}};
		}
		return {1, false, true, {}};
	}
	if (BoundComparisonExpression::IsComparison(expr)) {
		auto &comparison = expr.Cast<BoundFunctionExpression>();
		auto type = comparison.GetExpressionType();
		auto &left = BoundComparisonExpression::Left(comparison);
		auto &right = BoundComparisonExpression::Right(comparison);
		optional_ptr<const Expression> constant;
		if (IsLocalColumnRef(left) && right.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
			constant = right;
		} else if (IsLocalColumnRef(right) && left.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
			constant = left;
		}
		if (constant) {
			if (constant->Cast<BoundConstantExpression>().GetValue().IsNull()) {
				if (type == ExpressionType::COMPARE_DISTINCT_FROM) {
					return {1, false, false, {}};
				}
				if (type != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
					// Null-intolerant comparisons against NULL never evaluate to TRUE.
					return {0, true, false, {}};
				}
				return {1, false, true, {}};
			}
			if (type == ExpressionType::COMPARE_NOTEQUAL || type == ExpressionType::COMPARE_DISTINCT_FROM) {
				return {1, false, false, {}};
			}
		}
	}
	ColumnBinding target_binding;
	if (TryGetSingleColumnBinding(expr, target_binding)) {
		auto bound =
		    RelationStatisticsHelper::EstimateDirectFilterDomain(expr, target_binding, NumericLimits<idx_t>::Maximum());
		if (bound.IsValid()) {
			if (bound.GetIndex() == 0) {
				return {0, true, false, {}};
			}
			auto entry = domains.find(target_binding);
			if (entry != domains.end()) {
				auto &info = entry->second;
				CardinalityValue bound_value(bound.GetIndex());
				if (bound_value >= info.domain) {
					return {1, false, false, {}};
				}
				return {bound_value / info.domain, true, false, {info.canonical_binding}, info.relation_idx,
				        info.relation_cardinality};
			}
		}
	}
	return {1, false, true, {}};
}

vector<ResidualFilterEstimate>
ResidualFilterEstimator::EstimateBatch(const vector<optional_ptr<const Expression>> &expressions) const {
	const CardinalityValue default_selectivity(RelationStatisticsHelper::DEFAULT_SELECTIVITY);
	vector<ResidualFilterEstimate> results(expressions.size(), ResidualFilterEstimate {default_selectivity, false});
	if (expressions.empty()) {
		return results;
	}
	vector<ExpressionEstimate> estimates(expressions.size());
	column_binding_map_t<idx_t> evidence_counts;
	for (idx_t i = 0; i < expressions.size(); i++) {
		auto expr = expressions[i];
		if (!expr) {
			continue;
		}
		if (expr->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT &&
		    expr->GetReturnType() == LogicalType::BOOLEAN) {
			auto &value = expr->Cast<BoundConstantExpression>().GetValue();
			bool is_true = !value.IsNull() && value.GetValue<bool>();
			results[i] = {CardinalityValue(is_true ? 1 : 0), !is_true};
			continue;
		}
		estimates[i] = EstimateExpression(*expr, 0);
		if (estimates[i].has_evidence && !estimates[i].selectivity.IsZero()) {
			for (auto &col : estimates[i].evidence_columns) {
				evidence_counts[col]++;
			}
		}
	}
	for (idx_t i = 0; i < expressions.size(); i++) {
		auto expr = expressions[i];
		if (!expr || (expr->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT &&
		              expr->GetReturnType() == LogicalType::BOOLEAN)) {
			continue;
		}
		auto &est = estimates[i];
		if (est.selectivity.IsZero()) {
			results[i] = {CardinalityValue(0), true};
			continue;
		}
		auto effective = est.Effective(default_selectivity);
		bool has_overlap = false;
		if (est.has_evidence) {
			for (auto &col : est.evidence_columns) {
				if (evidence_counts[col] > 1) {
					has_overlap = true;
					break;
				}
			}
		}
		// Only refine selective non-overlapping residuals; all other cases retain the default estimate.
		results[i].selectivity =
		    (!est.has_evidence || has_overlap) ? default_selectivity : MinValue(default_selectivity, effective);
	}
	return results;
}

} // namespace duckdb
