/**
 * @file advection2D.cc
 * @brief Defines advection2D class
 */

#include "advection2D.h"

/**
 * @brief Constructor with order of polynomial approx as arg
 */
advection2D::advection2D(const int order)
: mapping(), fe(order), dof_handler(triang) // @suppress("Symbol is not resolved")
{}

/**
 * @brief Sets up the system
 */
void advection2D::setup_system()
{
        // initialise the triang variable
        GridGenerator::hyper_cube(triang);
        triang.refine_global(5); // 2^5=32 cells in each direction

        dof_handler.distribute_dofs(fe);

        // no system_matrix because the solution is updated cell wise
        g_solution.reinit(dof_handler.n_dofs());
        gold_solution.reinit(dof_handler.n_dofs());
        l_rhs.reinit(fe.dofs_per_cell);
}

/**
 * @brief Assembles the system using MeshWorker
 */
void advection2D::assemble_system()
{
        
}
