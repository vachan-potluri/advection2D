/**
 * @file advection2D.cc
 * @brief Defines advection2D class
 */

#include "advection2D.h"

/**
 * @brief Constructor with order of polynomial approx as arg
 * 
 * Based on order, face_first_dof and face_dof_increment containers are set here. See
 * https://www.dealii.org/current/doxygen/deal.II/structGeometryInfo.html and
 * https://www.dealii.org/current/doxygen/deal.II/classFE__DGQ.html for face and dof ordering
 * respectively in a cell. According to GeometryInfo, the direction of face lines is along the
 * positive axes. See DG notes dated 24-09-19.
 * 
 * Eg: for order=2, on 1-th face, the first cell dof is 2 and the next dof is obtained after
 * increment of 3
 */
advection2D::advection2D(const uint order)
: mapping(), fe(order), fe_face(order), dof_handler(triang),
        face_first_dof{0, order, 0, (order+1)*order},
        face_dof_increment{order+1, order+1, 1, 1}
{}

/**
 * @brief Sets up the system
 * 
 * 1. Mesh is setup and stored in triang
 * 2. dof_handler is linked to fe
 * 3. Global solution and local rhs sizes are set
 * 4. Cell user indices are set (for owner/neighbor distinction)
 * 5. Sizes of stiffness and lifting matrix containers are set
 */
void advection2D::setup_system()
{
        deallog << "Setting up the system" << std::endl;
        // initialise the triang variable
        GridGenerator::hyper_cube(triang);
        triang.refine_global(5); // 2^5=32 cells in each direction, total length 1m

        // set dof_handler
        dof_handler.distribute_dofs(fe);

        // no system_matrix because the solution is updated cell wise
        g_solution.reinit(dof_handler.n_dofs());
        gold_solution.reinit(dof_handler.n_dofs());
        l_rhs.reinit(fe.dofs_per_cell);

        // set user flags for cell
        // for a face, cell with lower user index will be treated owner
        // is this reqd? can't we just use cell->index()?
        // uint i=0;
        // for(auto &cell: dof_handler.active_cell_iterators()){
        //         cell->set_user_index(i++);
        // } // loop over cells

        // set sizes of stiffness and lifting matrix containers
        stiff_mats.reserve(triang.n_active_cells());
        lift_mats.reserve(triang.n_active_cells());
}

/**
 * @brief Assembles the system
 * 
 * Calculating mass and differentiation matrices is as usual. Each face will have its own flux
 * matrix. The containers face_first_dof and face_dof_increment are used to map face-local dof index
 * to cell dof index.
 */
void advection2D::assemble_system()
{
        deallog << "Assembling system ... " << std::flush;
        // allocate all local matrices
        FullMatrix<double> l_mass(fe.dofs_per_cell),
                l_mass_inv(fe.dofs_per_cell),
                l_diff(fe.dofs_per_cell),
                l_flux(fe.dofs_per_cell),
                temp(fe.dofs_per_cell); // initialise with square matrix size
        QGauss<2> cell_quad_formula(fe.degree+1); // (N+1) gauss quad for cell
        QGauss<1> face_quad_formula(fe.degree+1); // for face
        FEValues<2> fe_values(fe, cell_quad_formula,
                update_values | update_gradients | update_JxW_values | update_quadrature_points);
        FEFaceValues<2> fe_face_values(fe, face_quad_formula,
                update_values | update_JxW_values | update_quadrature_points);
        
        uint i, j, i_face, j_face, qid, face_id;
        // compute mass and diff matrices
        for(auto &cell: dof_handler.active_cell_iterators()){
                fe_values.reinit(cell);
                l_mass = 0;
                l_diff = 0;
                for(qid=0; qid<fe_values.n_quadrature_points; qid++){
                        for(i=0; i<fe.dofs_per_cell; i++){
                                for(j=0; j<fe.dofs_per_cell; j++){
                                        l_mass(i,j) += fe_values.shape_value(i, qid) *
                                                fe_values.shape_value(j, qid) *
                                                fe_values.JxW(qid);
                                        l_diff(i,j) += fe_values.shape_grad(i, qid) *
                                                wind(fe_values.quadrature_point(qid)) *
                                                fe_values.shape_value(j, qid) *
                                                fe_values.JxW(qid);
                                } // inner loop cell shape fns
                        } // outer loop cell shape fns
                } // loop over cell quad points
                l_mass_inv.invert(l_mass);
                l_mass_inv.mmult(temp, l_diff); // store mass_inv * diff into temp
                stiff_mats.emplace_back(temp);

                // each face will have a separate flux matrix
                for(face_id=0; face_id<GeometryInfo<2>::faces_per_cell; face_id++){
                        fe_face_values.reinit(cell, face_id);
                        l_flux = 0;
                        for(qid=0; qid<fe_face_values.n_quadrature_points; qid++){
                                for(i_face=0; i_face<fe_face.dofs_per_face; i_face++){
                                        for(j_face=0; j_face<fe_face.dofs_per_face; j_face++){
                                                // mapping
                                                i = face_first_dof[face_id] +
                                                        i_face*face_dof_increment[face_id];
                                                j = face_first_dof[face_id] +
                                                        j_face*face_dof_increment[face_id];
                                                l_flux(i,j) +=
                                                        fe_face_values.shape_value(i, qid) *
                                                        fe_face_values.shape_value(j, qid) *
                                                        fe_face_values.JxW(qid);
                                        } // inner loop over face shape fns
                                } // outer loop over face shape fns
                        } // loop over face quad points
                        l_mass_inv.mmult(temp, l_flux);
                        lift_mats[cell->index()][face_id] = temp;
                }// loop over faces
        }// loop over cells
        deallog << "Completed assembly" << std::endl;
}

/**
 * @brief Sets initial condition
 * 
 * Since nodal basis is being used, initial condition is easy to set
 */
void advection2D::set_IC()
{
        std::vector<uint> cur_global_dofs;
        cur_global_dofs.reserve(fe.dofs_per_cell);
        for(auto &cell: dof_handler.active_cell_iterators()){
                cell->get_dof_indices(cur_global_dofs);
                
        }
}

/**
 * @brief Prints stifness and the 4 lifting matrices of 0-th element
 */
void advection2D::print_matrices()
{
        deallog << "Stiffness matrix" << std::endl;
        stiff_mats[0].print(deallog, 10, 2);
        for(uint i=0; i<GeometryInfo<2>::faces_per_cell; i++){
                deallog << "Lifting matrix, face " << i << std::endl;
                lift_mats[0][i].print(deallog, 15, 4);
        }
}



// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
// Test function
// # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#ifdef DEBUG
void advection2D::test()
{
        deallog << "---------------------------------------------" << std::endl;
        deallog << "Testing advection2D class" << std::endl;
        deallog << "---------------------------------------------" << std::endl;
        advection2D problem(1);
        problem.setup_system();
        problem.assemble_system();
        problem.print_matrices();
}
#endif
