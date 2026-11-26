//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/relation_statistics/residual_filter_selectivity.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/optimizer/relation_statistics/relation_statistics.hpp"
#include "duckdb/planner/column_binding_map.hpp"

namespace duckdb {

class Expression;

struct ResidualFilterEstimate {
	CardinalityValue selectivity = 1;
	bool proven_empty = false;
};

//! Estimates selectivities of boolean residual predicates from conditioned leaf column domains.
class ResidualFilterEstimator {
public:
	//! Leaf relations must still carry their original column bindings; equivalence_roots maps normalized
	//! (relation_idx, column_idx) bindings to a shared representative per INNER equality class.
	explicit ResidualFilterEstimator(const vector<RelationStats> &relations,
	                                 const column_binding_map_t<ColumnBinding> &equivalence_roots = {});

public:
	//! Estimate a batch of residual predicates with cross-predicate shared-column overlap fallback.
	vector<ResidualFilterEstimate> EstimateBatch(const vector<optional_ptr<const Expression>> &expressions) const;

private:
	struct ColumnDomain {
		idx_t relation_idx;
		CardinalityValue relation_cardinality;
		CardinalityValue domain;
		ColumnBinding canonical_binding;
	};

	struct ExpressionEstimate;

private:
	ExpressionEstimate EstimateExpression(const Expression &expr, idx_t depth) const;

private:
	column_binding_map_t<ColumnDomain> domains;
};

} // namespace duckdb
