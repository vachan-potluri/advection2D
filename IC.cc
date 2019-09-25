/**
 * @file IC.cc
 * @brief Contains definition for initial condition
 */

#include "IC.h"

void IC::value_list(const std::vector<Point<2>> &points,
        std::vector<double> &values, const uint component) const
{
        for(double &cur_value: values) cur_value=0.0; // constant initial condition
}