/**
 * @file advection2D.cc
 * @brief Defines advection2D class
 */

#include "advection2D.h"

/**
 * @brief Constructor with @p order of polynomial approx as arg
 * 
 * advection2D::mapping, advection2D::fe and advection2D::fe_face are initialised.
 * advection2D::dof_handler is associated to advection2D::triang.
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
 * 1. Mesh is setup and stored in advection2D::triang
 * 2. advection2D::dof_handler is linked to advection2D::fe
 * 3. advection2D::g_solution and advection2D::l_rhs sizes are set
 * 4. Sizes of advection2D::stiff_mats, advection2D::lift_mats and advection2D::l_rhs containers are
 * set
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

        l_rhs.reserve(triang.n_active_cells());
        for(auto &cur_rhs: l_rhs) cur_rhs.reinit(fe.dofs_per_cell);
}

/**
 * @brief Assembles the system
 * 
 * Calculating mass and differentiation matrices is as usual. Each face will have its own flux
 * matrix. The containers advection2D::face_first_dof and advection2D::face_dof_increment are used
 * to map face-local dof index to cell dof index.
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
                deallog << "Assembling cell " << cell->index() << std::endl;
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
                stiff_mats[cell->index()] = temp;

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
 * Since nodal basis is being used, initial condition is easy to set. interpolate function of
 * VectorTools namespace is used with IC class and advection2D::g_solution. See IC::value()
 */
void advection2D::set_IC()
{
        VectorTools::interpolate(dof_handler, IC(), g_solution);
}

/**
 * @brief Boundary ids are set here
 * 
 * @f$x=0@f$ forms boundary 0 with @f$\phi@f$ value prescribed as @f$1@f$<br/>
 * @f$y=0@f$ forms boundary 1 with @f$\phi@f$ value prescribed as @f$0@f$<br/>
 * @f$x=1 \bigcup y=1@f$ forms boundary 2 with zero gradient
 * @note Ghost cell approach will be used
 * @todo Check this function
 */
void advection2D::set_boundary_ids()
{
        for(auto &cell: dof_handler.active_cell_iterators()){
                for(uint face_id=0; face_id<GeometryInfo<2>::faces_per_cell; face_id++){
                        if(cell->face(face_id)->at_boundary()){
                                Point<2> fcenter = cell->face(face_id)->center(); // face center
                                if(fabs(fcenter(0)) < 1e-6) cell->face(face_id)->set_boundary_id(0);
                                if(fabs(fcenter(1)) < 1e-6) cell->face(face_id)->set_boundary_id(1);
                                else cell->face(face_id)->set_boundary_id(2);
                        }
                } // loop over faces
        } // loop over cells
}

/**
 * @brief Updates solution with the given @p time_step
 * 
 * Algorithm:
 * - For every cell:
 *   - For every face:
 *     - Get neighbor id
 *     - if neighbor id > cell id, continue
 *     - else:
 *       - Get face id wrt owner and neighbor (using neighbor_of_neighbor)
 *       - Get global dofs on owner and neighbor
 *       - Using face ids and global dofs of owner and neighbor, get global dofs on this face on
 * owner and neighbor side
 *       - Compute the numerical flux
 *       - Use lifting matrices to update owner and neighbor rhs
 * 
 * <code>cell->get_dof_indices()</code> will return the dof indices in the order shown in
 * https://www.dealii.org/current/doxygen/deal.II/classFE__DGQ.html. This fact is mentioned in
 * https://www.dealii.org/current/doxygen/deal.II/classDoFCellAccessor.html
 * 
 * @pre @p time_step must be a stable one, any checks on this value are not done
 */
void advection2D::update(const double time_step)
{
        uint face_id, face_id_neighbor, i;
        std::vector<uint> dof_ids(fe.dofs_per_cell), dof_ids_neighbor(fe.dofs_per_cell);
        double phi, phi_neighbor; // owner and neighbor side values of phi
        Vector<double> normal_flux(fe_face.dofs_per_face); // the normal num flux vector for a face
        Point<2> dof_loc;
        Tensor<1,2> normal;

        for(auto &cell: dof_handler.active_cell_iterators()){
                for(face_id=0; face_id<GeometryInfo<2>::faces_per_cell; face_id++){
                        if(cell->neighbor(face_id)->index() > cell->index()) continue;
                        else{
                                // get normal
                                face_id_neighbor = cell->neighbor_of_neighbor(face_id);
                                cell->get_dof_indices(dof_ids);
                                cell->neighbor(face_id)->get_dof_indices(dof_ids_neighbor);
                                for(i=0; i<fe_face.dofs_per_face; i++){
                                        phi = gold_solution[
                                                face_first_dof[face_id] +
                                                i*face_dof_increment[face_id]];
                                        phi_neighbor = gold_solution[
                                                face_first_dof[face_id_neighbor] +
                                                i*face_dof_increment[face_id_neighbor]];
                                } // loop over face dofs
                        } // loop over faces
                } // loop over cells
        }
}

/**
 * @brief Prints stifness and the 4 lifting matrices of 0-th element
 */
void advection2D::print_matrices() const
{
        deallog << "Stiffness matrix" << std::endl;
        stiff_mats[0].print(deallog, 10, 2);
        for(uint i=0; i<GeometryInfo<2>::faces_per_cell; i++){
                deallog << "Lifting matrix, face " << i << std::endl;
                lift_mats[0][i].print(deallog, 15, 4);
        }
}

/**
 * @brief Outputs the global solution in vtk format taking the filename as argument
 */
void advection2D::output(const std::string &filename) const
{
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(g_solution, "phi");

        data_out.build_patches();

        std::ofstream ofile(filename);
        data_out.write_vtk(ofile);
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
        problem.set_IC();
        problem.set_boundary_ids();
        problem.output(std::string("initial_condition.vtk"));
}
#endif
