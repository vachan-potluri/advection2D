/**
 * @file IC.h
 * @brief Contains declaration for initial condition
 */

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include "common.h"

class IC: public Function<2>
{
        public:
        IC() = default;
        virtual void value_list(const std::vector<Point<2>> &points,
                std::vector<double> &values, const uint component=0) const override;
};