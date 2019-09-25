/**
 * @file IC.cc
 * @brief Contains definition for initial condition
 */

#include "IC.h"

void IC::value_list(const std::vector<Point<2>> &points,
        std::vector<double> &values,
        const uint component) const
{
        auto point_iter = points.cbegin();
        auto val_iter = values.begin();
        while(point_iter != points.cend()){
                *val_iter = 0.0; // constant initial condition
                point_iter++;
                val_iter++;
        }
}