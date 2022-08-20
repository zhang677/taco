#ifndef TACO_MODE_FORMAT_COORD_H
#define TACO_MODE_FORMAT_COORD_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class CoordModeFormat : public ModeFormatImpl {
public:
  using ModeFormatImpl::getInsertCoord;

  CoordModeFormat();
  ~CoordModeFormat() override {}
  ModeFormat copy(std::vector<ModeFormat::Property> properties) const override;
  std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode,int level) const override;
protected:
  ir::Stmt getTryInsert(ir::Expr insertFail, ir::Expr accumulator, ir::Expr accumulator_size
  , ir::Expr accumulator_capacity, ir::Expr crds, ir::Expr expression, Datatype type) const;
};
}

#endif
