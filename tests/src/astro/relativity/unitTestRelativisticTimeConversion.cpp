/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "tudat/basics/testMacros.h"

#include "tudat/math/basic/linearAlgebra.h"
#include "tudat/astro/basic_astro/physicalConstants.h"
#include "tudat/astro/basic_astro/orbitalElementConversions.h"

#include "tudat/io/basicInputOutput.h"
#include "tudat/interface/spice/spiceInterface.h"
#include "tudat/math/integrators/rungeKuttaCoefficients.h"

#include <tudat/simulation/simulation.h>
#include "tudat/astro/basic_astro/timeConversions.h"
#include "tudat/interface/sofa/earthOrientation.h"
#include "tudat/astro/ephemerides/keplerEphemeris.h"

#include "tudat/astro/relativity/relativisticTimeConversion.h"
#include "tudat/interface/sofa/sofaTimeConversions.h"
#include "tudat/io/readInpopEphemerisFile.h"
#include "tudat/math/integrators/createNumericalIntegrator.h"
#include "tudat/simulation/environment_setup/defaultBodies.h"
#include "tudat/simulation/environment_setup/createBodies.h"
#include "tudat/simulation/environment_setup/createGroundStations.h"
#include "tudat/interface/spice/spiceEphemeris.h"

#include "tudat/simulation/environment_setup/createRelativisticTimeConverter.h"

#include "tudat/simulation/simulation.h"
#include "tudat/basics/timeType.h"
#include "tudat/simulation/environment_setup/body.h"
#include "tudat/math/basic/leastSquaresEstimation.h"



namespace tudat
{

namespace unit_tests
{

using namespace tudat::simulation_setup;
using namespace tudat::propagators;
using namespace tudat::numerical_integrators;
using namespace tudat::orbital_element_conversions;
using namespace tudat::basic_mathematics;
using namespace tudat::unit_conversions;
using namespace tudat::input_output;


BOOST_AUTO_TEST_SUITE( test_RelativisticConversions )

BOOST_AUTO_TEST_CASE( test_tcb_to_tcg_conversion )
{

    std::string spiceKernelsPath = paths::getSpiceKernelPath( );
    std::string textKernelsPath = paths::getSpiceKernelPath( ) + "/inpop19a_TDB_m100_p100_asc";

    //Load spice kernels.
    std::string kernelsPath = paths::getSpiceKernelPath( );
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop19a_TDB_m100_p100_spice.tpc");
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop19a_TDB_m100_p100_spice.bsp");
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop19a_TDB_m100_p100_spice.bpc");
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop19a_TDB_m100_p100_spice.tf");

    // Map from SPICE ID string to body name
    std::map< std::string, std::string > bodyIdToName = {
        { "1", "Mercury" },
        { "2", "Venus" },
        { "3", "Earth" },
        { "4", "Mars" },
        { "5", "Jupiter" },
        { "6", "Saturn" },
        { "7", "Uranus" },
        { "8", "Neptune" },
        { "9", "Pluto" },
        { "10", "Sun" },
        { "301", "Moon" },
        { "2000001", "Ceres" },
        { "2000002", "Pallas" },
        { "2000004", "Vesta" }
        // { "2000003", "Body2000003" },
        // { "2000006", "Body2000006" },
        // { "2000007", "Body2000007" },
        // { "2000008", "Body2000008" },
        // { "2000009", "Body2000009" },
        // { "2000010", "Body2000010" }
    };

    SystemOfBodies bodies;
    for ( const auto& [id, name] : bodyIdToName )
    {
        std::shared_ptr< Body > body = std::make_shared< Body >();
        double gm = spice_interface::getBodyGravitationalParameter( id ) / ( 1.0 - physical_constants::LB_TIME_RATE_TERM );
        body->setGravityFieldModel( std::make_shared< gravitation::GravityFieldModel >( gm ) );
        bodies.addBody( body, name );
    }

    // Specify initial time
    double initialEphemerisTime = -365.25 * 86400.0 * 2.0;
    double finalEphemerisTime = 365.25 * 86400.0 * 2.0;
    double maximumTimeStep = 3600.0;
    double numberOfTimeStepBuffer = 6.0;
    double buffer = numberOfTimeStepBuffer * maximumTimeStep;
    std::string centralBody = "Earth"; //Earth

    bodies.at( "Sun" )->setEphemeris( createInpopEphemerisFromFiles(
                                        textKernelsPath + "/inpop19a_TDB_m100_p100_asc_pos_Sun.asc",
                                        textKernelsPath + "/inpop19a_TDB_m100_p100_asc_vel_Sun.asc" ) );    
    bodies.at( "Mercury" )->setEphemeris( createInpopEphemerisFromFiles(
                                            textKernelsPath + "/inpop19a_TDB_m100_p100_asc_pos_Mer.asc",
                                            textKernelsPath + "/inpop19a_TDB_m100_p100_asc_vel_Mer.asc" ) );
    bodies.at( "Venus" )->setEphemeris( createInpopEphemerisFromFiles(
                                          textKernelsPath + "/inpop19a_TDB_m100_p100_asc_pos_Ven.asc",
                                          textKernelsPath + "/inpop19a_TDB_m100_p100_asc_vel_Ven.asc" ) );
    bodies.at( "Earth" )->setEphemeris( createInpopEphemerisFromFiles(
                                          textKernelsPath + "/inpop19a_TDB_m100_p100_asc_pos_Ear.asc",
                                          textKernelsPath + "/inpop19a_TDB_m100_p100_asc_vel_Ear.asc" ) );
    bodies.at( "Moon" )->setEphemeris( createInpopEphemerisFromFiles(
                                         textKernelsPath + "/inpop19a_TDB_m100_p100_asc_pos_Moo.asc",
                                         textKernelsPath + "/inpop19a_TDB_m100_p100_asc_vel_Moo.asc",
                                         basic_astrodynamics::JULIAN_DAY_ON_J2000, 1 ) );
    bodies.at( "Mars" )->setEphemeris( createInpopEphemerisFromFiles(
                                         textKernelsPath + "/inpop19a_TDB_m100_p100_asc_pos_Mar.asc",
                                         textKernelsPath + "/inpop19a_TDB_m100_p100_asc_vel_Mar.asc" ) );
    bodies.at( "Jupiter" )->setEphemeris( createInpopEphemerisFromFiles(
                                            textKernelsPath + "/inpop19a_TDB_m100_p100_asc_pos_Jup.asc",
                                            textKernelsPath + "/inpop19a_TDB_m100_p100_asc_vel_Jup.asc" ) );
    bodies.at( "Saturn" )->setEphemeris( createInpopEphemerisFromFiles(
                                           textKernelsPath + "/inpop19a_TDB_m100_p100_asc_pos_Sat.asc",
                                           textKernelsPath + "/inpop19a_TDB_m100_p100_asc_vel_Sat.asc" ) );
    bodies.at( "Uranus" )->setEphemeris( createInpopEphemerisFromFiles(
                                           textKernelsPath + "/inpop19a_TDB_m100_p100_asc_pos_Ura.asc",
                                           textKernelsPath + "/inpop19a_TDB_m100_p100_asc_vel_Ura.asc" ) );
    bodies.at( "Neptune" )->setEphemeris( createInpopEphemerisFromFiles(
                                            textKernelsPath + "/inpop19a_TDB_m100_p100_asc_pos_Nep.asc",
                                            textKernelsPath + "/inpop19a_TDB_m100_p100_asc_vel_Nep.asc" ) );
    bodies.at( "Pluto" )->setEphemeris( createInpopEphemerisFromFiles(
                                          textKernelsPath + "/inpop19a_TDB_m100_p100_asc_pos_Plu.asc",
                                          textKernelsPath + "/inpop19a_TDB_m100_p100_asc_vel_Plu.asc" ) );

    spice_interface::loadSpiceKernelInTudat( spiceKernelsPath + "/codes_300ast_20100725.bsp");
    spice_interface::loadSpiceKernelInTudat( spiceKernelsPath + "/codes_300ast_20100725.tf");

    bodies.at( "Ceres" )->setEphemeris( createTabulatedEphemerisFromSpice(
                                          "Ceres", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    bodies.at( "Pallas" )->setEphemeris( createTabulatedEphemerisFromSpice(
                                          "Pallas", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    bodies.at( "Vesta" )->setEphemeris( createTabulatedEphemerisFromSpice(
                                            "Vesta", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Body2000003" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Body2000003", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Body2000006" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Body2000006", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Body2000007" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Body2000007", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Body2000008" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Body2000008", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Body2000009" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Body2000009", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Body2000010" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Body2000010", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );

    setGlobalFrameBodyEphemerides( bodies.getMap(), "SSB", "ECLIPJ2000" );

    std::vector< std::string > externalBodies;
    for ( const auto& [id, name] : bodyIdToName )
    {
        if ( name != centralBody )
        {
            externalBodies.push_back( name );
        }
    }

    double startTime = initialEphemerisTime;
    double endTime = finalEphemerisTime;
    double timeStep = 6000.0;

    std::shared_ptr< numerical_integrators::IntegratorSettings< double > > integratorSettings = numerical_integrators::rungeKutta4Settings( timeStep );
    std::shared_ptr< PropagationTimeTerminationSettings > terminationSettings = std::make_shared< propagators::PropagationTimeTerminationSettings >( endTime );
                
    std::shared_ptr< propagators::SecondOrderBodyCenteredRelativisticTimeConverterSettings<double, double > > properTimeEquationSettings =
            std::make_shared< propagators::SecondOrderBodyCenteredRelativisticTimeConverterSettings<double, double > >(  
                centralBody, externalBodies, startTime, integratorSettings, terminationSettings );

    SingleArcDynamicsSimulator< > timeEquationPropagator = SingleArcDynamicsSimulator< >( bodies, integratorSettings, properTimeEquationSettings );


    std::string timeDifferenceFileName = textKernelsPath + "inpop19a_TDB_m100_p100_asc_pos_TCG.asc" ;
    std::shared_ptr< interpolators::OneDimensionalInterpolator< double, long double > > timeEphemerisInterpolator =
            input_output::createLongInpopTimeEphemerisInterpolator( timeDifferenceFileName );
    std::map< double, double > timeDifferences2;

    int counter = 0;
    long double initialDifference = 0.0L;
    long double rawDifference;

    std::shared_ptr< TimeEphemeris > earthTimeEphemeris = bodies.at( "Earth" )->getTimeScaleConverter( );
    std::function< double( const double ) > timeDifferenceFunction =
            earthTimeEphemeris->getTimeDifferenceFunction( basic_astrodynamics::barycentric_coordinate_time_scale, basic_astrodynamics::body_centered_coordinate_time_scale, "" );
    double testTimeStep = 7100.0; //To prevent excessive resonance with integration step.
    double currentTime = initialEphemerisTime + 5.0 * timeStep;
    while( currentTime < finalEphemerisTime - 5.0 * timeStep )
    {
        rawDifference = timeEphemerisInterpolator->interpolate( currentTime )-
                static_cast< long double >( timeDifferenceFunction( currentTime ) );

        if( counter == 0 )
        {
            initialDifference = rawDifference;
        }

        timeDifferences2[ currentTime ] = static_cast< double >( rawDifference - initialDifference );
        //std::cout<<timeDifferenceFunction( timeMap->first )<<" "<<timeMap->second<<" "<<
        //           timeDifferenceFunction( timeMap->first )-timeMap->second<<std::endl;
        currentTime += testTimeStep;
    }

    Eigen::VectorXd timesVector = utilities::convertStlVectorToEigenVector(
                utilities::createVectorFromMapKeys( timeDifferences2 ) );
    Eigen::VectorXd resultDifferenceVector = utilities::convertStlVectorToEigenVector(
                utilities::createVectorFromMapValues( timeDifferences2 ) );
    
    std::vector<double> dummy = { 0.0, 1.0 };
    Eigen::VectorXd trendFit = linear_algebra::getLeastSquaresPolynomialFit(
    timesVector, resultDifferenceVector, dummy );

    BOOST_CHECK_SMALL( std::fabs( trendFit[ 1 ] ), 2.0E-18 );

    Eigen::VectorXd resultDifferenceWithoutTrend = resultDifferenceVector - (
        Eigen::VectorXd::Constant( resultDifferenceVector.rows( ), trendFit[ 0 ] ) + trendFit[ 1 ] * timesVector );


    double maximumDifference = resultDifferenceWithoutTrend.maxCoeff( );
    double minimumDifference = resultDifferenceWithoutTrend.minCoeff( );

    BOOST_CHECK_SMALL( maximumDifference, 5.0E-12 );
    BOOST_CHECK_SMALL( std::fabs( minimumDifference ), 5.0E-12 );

    //std::cout<<minimumDifference<<" "<<maximumDifference<<std::endl;

    //output::writeDoubleMapToFile( timeDifferences2, "tcgMinusTcbInpop2.dat" );

}

BOOST_AUTO_TEST_SUITE_END( )

}  // namespace unit_tests

}  // namespace tudat