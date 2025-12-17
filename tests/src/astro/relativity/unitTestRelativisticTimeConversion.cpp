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
#include "tudat/math/basic/coordinateConversions.h"

#include "tudat/io/basicInputOutput.h"
#include "tudat/interface/spice/spiceInterface.h"
#include "tudat/math/integrators/rungeKuttaCoefficients.h"

#include <tudat/simulation/simulation.h>
#include "tudat/astro/basic_astro/timeConversions.h"
#include "tudat/interface/sofa/earthOrientation.h"
#include "tudat/astro/ephemerides/keplerEphemeris.h"
#include "tudat/astro/ephemerides/tleEphemeris.h"

#include "tudat/astro/relativity/relativisticTimeConversion.h"
#include "tudat/astro/relativity/metric.h"
#include "tudat/interface/sofa/sofaTimeConversions.h"
#include "tudat/io/readInpopEphemerisFile.h"
#include "tudat/math/integrators/createNumericalIntegrator.h"
#include "tudat/simulation/environment_setup/defaultBodies.h"
#include "tudat/simulation/environment_setup/createBodies.h"
#include "tudat/simulation/environment_setup/createGroundStations.h"
#include "tudat/interface/spice/spiceEphemeris.h"

#include "tudat/simulation/environment_setup/createRelativisticTimeConverter.h"
#include "tudat/simulation/environment_setup/createMetric.h"

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
using namespace tudat::basic_astrodynamics;


BOOST_AUTO_TEST_SUITE( test_RelativisticConversions )

BOOST_AUTO_TEST_CASE( test_tcb_to_tcg_conversion )
{

    std::string spiceKernelsPath = paths::getSpiceKernelPath( );
    std::string textKernelsPath = paths::getSpiceKernelPath( ) + "/inpop10e_TDB_m100_p100_asc";

    //Load spice kernels.
    std::string kernelsPath = paths::getSpiceKernelPath( );
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop10e_TDB_m100_p100_spice.tpc");
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop10e_TDB_m100_p100_spice.bsp");
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop10e_TDB_m100_p100_spice.bpc");
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop10e_TDB_m100_p100_spice.tf");

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
        { "2000004", "Vesta" },
        //{ "2000003", "Juno" },
        //{ "2000006", "Hebe" },
        //{ "2000007", "Iris" },
        //{ "2000008", "Flora" },
        //{ "2000009", "Metis" },
        //{ "2000010", "Hygiea" }
    };

    SystemOfBodies bodies;
    for ( const auto& idToNamePair : bodyIdToName )
    {
         const std::string& id = idToNamePair.first;
         const std::string& name = idToNamePair.second;
         std::shared_ptr< Body > body = std::make_shared< Body >();
         double gm = spice_interface::getBodyGravitationalParameter( id ); // / ( 1.0 - physical_constants::LB_TIME_RATE_TERM );
         body->setGravityFieldModel( std::make_shared< gravitation::GravityFieldModel >( gm ) );
         bodies.addBody( body, name );
     }

    // Specify initial time
    double initialEphemerisTime = -365.25 * 86400.0 * 2.0;
    double finalEphemerisTime = 365.25 * 86400.0 * 2.0; 
    double maximumTimeStep = 3600.0;
    double numberOfTimeStepBuffer = 6.0;
    double buffer = numberOfTimeStepBuffer * maximumTimeStep;
    std::string centralBody = "Earth"; 

    bodies.at( "Sun" )->setEphemeris( createInpopEphemerisFromFiles(
                                        textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_Sun.asc",
                                        textKernelsPath + "/inpop10e_TDB_m100_p100_asc_vel_Sun.asc" ) );    
    bodies.at( "Mercury" )->setEphemeris( createInpopEphemerisFromFiles(
                                            textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_Mer.asc",
                                            textKernelsPath + "/inpop10e_TDB_m100_p100_asc_vel_Mer.asc" ) );
    bodies.at( "Venus" )->setEphemeris( createInpopEphemerisFromFiles(
                                          textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_Ven.asc",
                                          textKernelsPath + "/inpop10e_TDB_m100_p100_asc_vel_Ven.asc" ) );
    bodies.at( "Earth" )->setEphemeris( createInpopEphemerisFromFiles(
                                          textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_Ear.asc",
                                          textKernelsPath + "/inpop10e_TDB_m100_p100_asc_vel_Ear.asc" ) );
    bodies.at( "Moon" )->setEphemeris( createInpopEphemerisFromFiles(
                                         textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_Moo.asc",
                                         textKernelsPath + "/inpop10e_TDB_m100_p100_asc_vel_Moo.asc",
                                         basic_astrodynamics::JULIAN_DAY_ON_J2000, 1 ) );
    bodies.at( "Mars" )->setEphemeris( createInpopEphemerisFromFiles(
                                         textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_Mar.asc",
                                         textKernelsPath + "/inpop10e_TDB_m100_p100_asc_vel_Mar.asc" ) );
    bodies.at( "Jupiter" )->setEphemeris( createInpopEphemerisFromFiles(
                                            textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_Jup.asc",
                                            textKernelsPath + "/inpop10e_TDB_m100_p100_asc_vel_Jup.asc" ) );
    bodies.at( "Saturn" )->setEphemeris( createInpopEphemerisFromFiles(
                                           textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_Sat.asc",
                                           textKernelsPath + "/inpop10e_TDB_m100_p100_asc_vel_Sat.asc" ) );
    bodies.at( "Uranus" )->setEphemeris( createInpopEphemerisFromFiles(
                                           textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_Ura.asc",
                                           textKernelsPath + "/inpop10e_TDB_m100_p100_asc_vel_Ura.asc" ) );
    bodies.at( "Neptune" )->setEphemeris( createInpopEphemerisFromFiles(
                                            textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_Nep.asc",
                                            textKernelsPath + "/inpop10e_TDB_m100_p100_asc_vel_Nep.asc" ) );
    bodies.at( "Pluto" )->setEphemeris( createInpopEphemerisFromFiles(
                                          textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_Plu.asc",
                                          textKernelsPath + "/inpop10e_TDB_m100_p100_asc_vel_Plu.asc" ) );

    spice_interface::loadSpiceKernelInTudat( spiceKernelsPath + "/codes_300ast_20100725.bsp");
    spice_interface::loadSpiceKernelInTudat( spiceKernelsPath + "/codes_300ast_20100725.tf");

    bodies.at( "Ceres" )->setEphemeris( createTabulatedEphemerisFromSpice(
                                          "Ceres", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    bodies.at( "Pallas" )->setEphemeris( createTabulatedEphemerisFromSpice(
                                          "Pallas", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    bodies.at( "Vesta" )->setEphemeris( createTabulatedEphemerisFromSpice(
                                            "Vesta", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Juno" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Juno", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Hebe" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Hebe", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Iris" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Iris", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Flora" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Flora", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Metis" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Metis", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );
    // bodies.at( "Hygiea" )->setEphemeris( createTabulatedEphemerisFromSpice(
    //                                         "Hygiea", initialEphemerisTime - buffer, finalEphemerisTime + buffer, 7200.0, "SSB", "ECLIPJ2000" ) );

    setGlobalFrameBodyEphemerides( bodies.getMap( ), "SSB", "ECLIPJ2000" );

    std::vector< std::string > externalBodies;
    for ( const auto& idToNamePair : bodyIdToName )
    {
        const std::string& name = idToNamePair.second;
        if ( name != centralBody )
        {
            externalBodies.push_back( name );
        }
    }

    double startTime = initialEphemerisTime;
    double endTime = finalEphemerisTime;
    double timeStep = 500; //6000.0;

    std::shared_ptr< numerical_integrators::IntegratorSettings< double > > integratorSettings =
            numerical_integrators::rungeKutta4Settings( timeStep );
    integratorSettings->initialTimeDeprecated_ = startTime;
    std::shared_ptr< PropagationTimeTerminationSettings > terminationSettings = std::make_shared< propagators::PropagationTimeTerminationSettings >( endTime );

    auto outputProcessingSettings = std::make_shared< SingleArcPropagatorProcessingSettings >(
                true,
                true,
                1,
                TUDAT_NAN,
                std::make_shared< PropagationPrintSettings >( true, false, 1 * 86400, 0, true, true, true, true, false, false ) );
    std::vector< std::shared_ptr< SingleDependentVariableSaveSettings > > dependentVariablesList{};

    std::shared_ptr< propagators::SecondOrderBodyCenteredRelativisticTimeConverterSettings<double, double > > properTimeEquationSettings =
            std::make_shared< propagators::SecondOrderBodyCenteredRelativisticTimeConverterSettings<double, double > >(  
                centralBody, externalBodies, startTime, integratorSettings, terminationSettings,
                ( std::map< std::string, std::pair< int, int > >( ) ),
                std::vector< std::string  >( ),
                &basic_astrodynamics::doDummyTimeConversion< double >,
                1.0,
                dependentVariablesList,
                outputProcessingSettings 
            );

    SingleArcDynamicsSimulator< > timeEquationPropagator = SingleArcDynamicsSimulator< >( bodies, properTimeEquationSettings );

    std::string timeDifferenceFileName = textKernelsPath + "/inpop10e_TDB_m100_p100_asc_pos_TT.asc";

    std::shared_ptr< interpolators::OneDimensionalInterpolator< double, long double > > timeEphemerisInterpolator =
            input_output::createLongInpopTimeEphemerisInterpolator( timeDifferenceFileName );
    std::map< double, double > timeDifferences2;

    int counter = 0;
    long double initialDifference = 0.0L;
    long double rawDifference;

    std::shared_ptr< Body > earth = bodies.getBody( "Earth" );
    std::shared_ptr< TimeEphemeris > earthTimeEphemeris = bodies.getBody( "Earth" )->getTimeScaleConverter( );

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
            counter += 1;
        }

        timeDifferences2[ currentTime ] = static_cast< double >( rawDifference - initialDifference );
        currentTime += testTimeStep;
    }

    input_output::writeDataMapToTextFile( timeDifferences2,
                                    "tcgMinusTcbInpop2.dat",
                                    tudat::paths::getTudatTestDataPath( ) + "", "", 16);

    Eigen::VectorXd timesVector = utilities::convertStlVectorToEigenVector(
                utilities::createVectorFromMapKeys( timeDifferences2 ) );
    Eigen::VectorXd resultDifferenceVector = utilities::convertStlVectorToEigenVector(
                utilities::createVectorFromMapValues( timeDifferences2 ) );
    
    std::vector<double> dummy = { 0.0, 1.0 };
    Eigen::VectorXd trendFit = linear_algebra::getLeastSquaresPolynomialFit( timesVector, resultDifferenceVector, dummy );

    BOOST_CHECK_SMALL( std::fabs( trendFit[ 1 ] ), 2.0E-18 );

    Eigen::VectorXd resultDifferenceWithoutTrend = resultDifferenceVector - (
        Eigen::VectorXd::Constant( resultDifferenceVector.rows( ), trendFit[ 0 ] ) + trendFit[ 1 ] * timesVector );

    double maximumDifference = resultDifferenceWithoutTrend.maxCoeff( );
    double minimumDifference = resultDifferenceWithoutTrend.minCoeff( );

    BOOST_CHECK_SMALL( maximumDifference, 5.0E-12 );
    BOOST_CHECK_SMALL( std::fabs( minimumDifference ), 5.0E-12 );

    // std::cout<< "maximumDifference" << maximumDifference << std::endl;
    
}


BOOST_AUTO_TEST_CASE( test_concatenated_conversions )
{

    std::string kernelsPath = paths::getSpiceKernelPath( );
    spice_interface::loadStandardSpiceKernels( );

    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop19a_TDB_m100_p100_spice.tpc");
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop19a_TDB_m100_p100_spice.bsp");
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop19a_TDB_m100_p100_spice.bpc");
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/inpop19a_TDB_m100_p100_spice.tf");
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/naif0012.tls");
    spice_interface::loadSpiceKernelInTudat( kernelsPath + "/pck00010.tpc");

    std::vector< std::string > bodyNames;
    bodyNames.push_back( "Earth" );
    bodyNames.push_back( "Sun" );
    bodyNames.push_back( "Moon" );
    bodyNames.push_back( "Jupiter" );
    bodyNames.push_back( "Saturn" );

    // Specify initial time
    double initialEphemerisTime = -365.25 * 86400.0 * 1.0;
    double finalEphemerisTime = 365.25 * 86400.0 * 1.0;
    double maximumTimeStep = 3600.0;
    double numberOfTimeStepBuffer = 6.0;
    double buffer = numberOfTimeStepBuffer * maximumTimeStep;
    std::string centralBody = "Earth";


    SystemOfBodies bodies;
    for( unsigned int i = 0; i < bodyNames.size( ); i++ )
    {
        if( bodyNames[ i ] != "Earth" )
        {
            std::shared_ptr< Body > body = std::make_shared< Body >( );
            body->setGravityFieldModel(
                        std::make_shared< gravitation::GravityFieldModel >(
                            spice_interface::getBodyGravitationalParameter( bodyNames[ i ] ) / ( 1.0 - physical_constants::LB_TIME_RATE_TERM ) ) );
            bodies.addBody( body, bodyNames[ i ] );
        }
    }

    
    std::shared_ptr< Body > earth = std::make_shared< Body >( );
    bodies.addBody( earth, "Earth" );
    earth->setShapeModel( createBodyShapeModel( getDefaultBodyShapeSettings( "Earth", initialEphemerisTime, finalEphemerisTime ), "Earth" ) );
    earth->setRotationalEphemeris( createRotationModel( getDefaultRotationModelSettings( "Earth", initialEphemerisTime, finalEphemerisTime ), "Earth" ) );

    std::shared_ptr< SphericalHarmonicsGravityFieldSettings > earthGravityFieldSettings =
            std::dynamic_pointer_cast< SphericalHarmonicsGravityFieldSettings >(
                getDefaultGravityFieldSettings( "Earth", initialEphemerisTime, finalEphemerisTime ) );
    earthGravityFieldSettings->resetAssociatedReferenceFrame( "IAU_Earth" );

    earth->setGravityFieldModel( createGravityFieldModel( earthGravityFieldSettings, "Earth", bodies ) );

    std::shared_ptr< Body > lro = std::make_shared< Body >( );

    Eigen::Vector6d lroInitialKeplerianElements;
    lroInitialKeplerianElements[ semiMajorAxisIndex ] = 2500.0E3;
    lroInitialKeplerianElements[ eccentricityIndex ] = 0.1;
    lroInitialKeplerianElements[ inclinationIndex ] = 0.375 * mathematical_constants::PI;
    lroInitialKeplerianElements[ argumentOfPeriapsisIndex ] = 0.0;
    lroInitialKeplerianElements[ longitudeOfAscendingNodeIndex ] = 0.0;
    lroInitialKeplerianElements[ trueAnomalyIndex ] = 0.0;

    lro->setEphemeris( std::make_shared< ephemerides::KeplerEphemeris >(
                           lroInitialKeplerianElements, initialEphemerisTime, spice_interface::getBodyGravitationalParameter( "Moon" ), "Moon"  ) );

    std::string textKernelsPath = paths::getSpiceKernelPath( ) + "/inpop19a_TCB_m100_p100_asc";

    bodies.addBody( lro, "LRO" );

    bodies.at( "Sun" )->setEphemeris( createInpopEphemerisFromFiles(
                                        textKernelsPath + "/inpop19a_TCB_m100_p100_asc_pos_Sun.asc",
                                        textKernelsPath + "/inpop19a_TCB_m100_p100_asc_vel_Sun.asc" ) );
    bodies.at( "Earth" )->setEphemeris( createInpopEphemerisFromFiles(
                                          textKernelsPath + "/inpop19a_TCB_m100_p100_asc_pos_Ear.asc",
                                          textKernelsPath + "/inpop19a_TCB_m100_p100_asc_vel_Ear.asc" ) );
    bodies.at( "Moon" )->setEphemeris( createInpopEphemerisFromFiles(
                                         textKernelsPath + "/inpop19a_TCB_m100_p100_asc_pos_Moo.asc",
                                         textKernelsPath + "/inpop19a_TCB_m100_p100_asc_vel_Moo.asc",
                                         basic_astrodynamics::JULIAN_DAY_ON_J2000, 1 ) );
    bodies.at( "Jupiter" )->setEphemeris( createInpopEphemerisFromFiles(
                                            textKernelsPath + "/inpop19a_TCB_m100_p100_asc_pos_Jup.asc",
                                            textKernelsPath + "/inpop19a_TCB_m100_p100_asc_vel_Jup.asc" ) );
    bodies.at( "Saturn" )->setEphemeris( createInpopEphemerisFromFiles(
                                           textKernelsPath + "/inpop19a_TCB_m100_p100_asc_pos_Sat.asc",
                                           textKernelsPath + "/inpop19a_TCB_m100_p100_asc_vel_Sat.asc" ) );
    setGlobalFrameBodyEphemerides( bodies.getMap( ), "SSB", "ECLIPJ2000" );

    // Create ground stations
    std::map< std::pair< std::string, std::string >, Eigen::Vector3d > groundStationsToCreate;
    groundStationsToCreate[ std::make_pair( "Earth", "Graz" ) ] = ( Eigen::Vector3d( ) << 4194511.7, 1162789.7, 4647362.5 ).finished( );
    groundStationsToCreate[ std::make_pair( "Earth", "Yarragadee" ) ] = ( Eigen::Vector3d( ) << -2389008, 5043332, -3078526 ).finished( );

    createGroundStations( bodies, groundStationsToCreate );
    
    std::vector< std::string > externalBodies;
    for( unsigned int i = 0; i < bodyNames.size( ); i++ )
    {
        if( bodyNames[ i ] != centralBody )
        {
            externalBodies.push_back( bodyNames[ i ] );
        }
    }

    double startTime = initialEphemerisTime;
    double endTime = finalEphemerisTime;
    double timeStep = 6000.0;

    std::shared_ptr< numerical_integrators::IntegratorSettings< double > > integratorSettings =
            numerical_integrators::rungeKutta4Settings( timeStep );
    std::shared_ptr< PropagationTimeTerminationSettings > terminationSettings = std::make_shared< propagators::PropagationTimeTerminationSettings >( endTime );

    std::vector< std::string > listOfPerturbingBodies{ "Earth",  "Moon",  "Sun", "Jupiter", "Saturn" };
    Eigen::Matrix< double, Eigen::Dynamic, 1 > initialRelativisticTimeState = Eigen::Matrix< double, Eigen::Dynamic, 1 >::Zero( 1 );

    auto outputProcessingSettings = std::make_shared< SingleArcPropagatorProcessingSettings >(
                true,
                true,
                1,
                TUDAT_NAN,
                std::make_shared< PropagationPrintSettings >( true, false, 1 * 86400, 0, true, true, true, true, false, false ) );
    std::vector< std::shared_ptr< SingleDependentVariableSaveSettings > > dependentVariablesList{};


    const std::vector< std::string > topocentricPerturbingBodies{ "Moon", "Sun", "Jupiter", "Saturn" };

    std::vector< std::shared_ptr< RelativisticTimeStatePropagatorSettings< double, double > > > bodyCentricToTopocentricConversionSettings;
    bodyCentricToTopocentricConversionSettings.push_back(
                std::make_shared< BodycenteredToTopocentricTimePropagatorSettings< double, double > >(
                    std::make_pair( "Earth", "Graz" ), 0, 4, 0, topocentricPerturbingBodies,
                    initialRelativisticTimeState, initialEphemerisTime, integratorSettings, terminationSettings,
                    dependentVariablesList, outputProcessingSettings ) );
    bodyCentricToTopocentricConversionSettings.push_back(
                std::make_shared< BodycenteredToTopocentricTimePropagatorSettings< double, double > >(
                    std::make_pair( "Earth", "Yarragadee" ), 0, 4, 0, topocentricPerturbingBodies,
                    initialRelativisticTimeState, initialEphemerisTime, integratorSettings, terminationSettings,
                    dependentVariablesList, outputProcessingSettings ) );

    std::map< std::string, std::shared_ptr< DirectRelativisticTimeConverterSettings<> > > relativisticConverterSettings;
    relativisticConverterSettings[ "LRO" ] = std::make_shared< DirectRelativisticTimeConverterSettings<> >(
                std::make_shared< propagators::FirstOrderBodycentricRelativisticTimePropagatorSettings< double, double > >(
                    "LRO", listOfPerturbingBodies, initialEphemerisTime, integratorSettings, terminationSettings ),
                integratorSettings ); 

    std::vector< std::string > listOfPerturbingBodies2{ "Moon",  "Sun", "Jupiter", "Saturn" };
    relativisticConverterSettings[ "Earth" ] = std::make_shared< DirectRelativisticTimeConverterSettings<> >(
                std::make_shared< propagators::SecondOrderBodyCenteredRelativisticTimeConverterSettings< double, double > >(
                    "Earth", listOfPerturbingBodies2, initialEphemerisTime, integratorSettings, terminationSettings ),
                integratorSettings,
                bodyCentricToTopocentricConversionSettings );

    setRelativisticTimeConverters( bodies, relativisticConverterSettings );

    std::shared_ptr< TimeEphemeris > earthTimeScaleConverter = earth->getTimeScaleConverter( );
    std::shared_ptr< TimeEphemeris > lroTimeScaleConverter = lro->getTimeScaleConverter( );

    BOOST_CHECK_EQUAL( ( lroTimeScaleConverter == NULL ), 0 );
    BOOST_CHECK_EQUAL( ( earthTimeScaleConverter == NULL ), 0 );

    BOOST_CHECK_SMALL( lroTimeScaleConverter->getTimeDifference(
                           body_centered_coordinate_time_scale, barycentric_coordinate_time_scale, initialEphemerisTime ),
                       std::numeric_limits< double >::epsilon( ) );
    BOOST_CHECK_SMALL( earthTimeScaleConverter->getTimeDifference(
                           body_centered_coordinate_time_scale, barycentric_coordinate_time_scale, initialEphemerisTime ),
                       std::numeric_limits< double >::epsilon( ) );
    BOOST_CHECK_SMALL( lroTimeScaleConverter->getTimeDifference(
                           barycentric_coordinate_time_scale, body_centered_coordinate_time_scale, initialEphemerisTime ),
                       std::numeric_limits< double >::epsilon( ) );
    BOOST_CHECK_SMALL( earthTimeScaleConverter->getTimeDifference(
                           barycentric_coordinate_time_scale, body_centered_coordinate_time_scale, initialEphemerisTime ),
                       std::numeric_limits< double >::epsilon( ) );


    BOOST_CHECK_SMALL( earthTimeScaleConverter->getTimeDifference(
                           body_centered_coordinate_time_scale, local_proper_time_scale, initialEphemerisTime, "Graz" ),
                       std::numeric_limits< double >::epsilon( ) );
    BOOST_CHECK_SMALL( earthTimeScaleConverter->getTimeDifference(
                           local_proper_time_scale, body_centered_coordinate_time_scale, initialEphemerisTime, "Graz" ),
                       std::numeric_limits< double >::epsilon( ) );

    std::shared_ptr< SecondOrderBodyCenteredRelativisticTimeConverterSettings< double, double > > directEarthTimeScaleConverter =
            std::make_shared< SecondOrderBodyCenteredRelativisticTimeConverterSettings< double, double > >(
                "Earth", listOfPerturbingBodies2, initialEphemerisTime, integratorSettings, terminationSettings );

    directEarthTimeScaleConverter->getOutputSettings( )->setIntegratedResult( true );

    // Get directly calculated map of tcg-tcb from tcb input (key)
    SingleArcDynamicsSimulator< > timeEquationPropagator = SingleArcDynamicsSimulator< >(
                bodies, directEarthTimeScaleConverter, true );

    std::map< double, Eigen::VectorXd > directTimeDifferencesVectors = timeEquationPropagator.getEquationsOfMotionNumericalSolution( );
    std::map< double, double > directTimeDifferences;
    for( std::map< double, Eigen::VectorXd >::iterator resultIterator = directTimeDifferencesVectors.begin( ); resultIterator !=
         directTimeDifferencesVectors.end( ); resultIterator++ )
    {
        directTimeDifferences[ resultIterator->first ] = resultIterator->second.x( );
    }

    // Create map of tcb-tcg from tcg input (key)
    std::map< double, double > directInverseTimeDifferences;
    for( std::map< double, double >::iterator differenceIterator = directTimeDifferences.begin( ); differenceIterator !=
         directTimeDifferences.end( ); differenceIterator++ )
    {
        directInverseTimeDifferences[ differenceIterator->first + differenceIterator->second ] = -differenceIterator->second;
    }

    // Get time difference functions from indirect calculator.
    std::function< double( const double ) > indirectDifferenceFunction = earthTimeScaleConverter->getTimeDifferenceFunction(
                barycentric_coordinate_time_scale, body_centered_coordinate_time_scale );
    std::function< double( const double ) > indirectInverseDifferenceFunction = earthTimeScaleConverter->getTimeDifferenceFunction(
                body_centered_coordinate_time_scale, barycentric_coordinate_time_scale );

    // Iterate over all directly calculated function values, and use indirect inverse function to check whether a zero difference results.
    Eigen::VectorXd forwardBackardTransformationResults =  Eigen::VectorXd( directTimeDifferences.size( ) );
    Eigen::VectorXd inverseForwardBackardTransformationResults =  Eigen::VectorXd( directTimeDifferences.size( ) );

    int counter = 0;
    double convertedValue = 0.0;
    for( std::map< double, double >::iterator differenceIterator = directTimeDifferences.begin( ); differenceIterator !=
         directTimeDifferences.end( ); differenceIterator++ )
    {
        convertedValue = indirectDifferenceFunction( differenceIterator->first );
        forwardBackardTransformationResults( counter ) = convertedValue - differenceIterator->second;
        counter++;
    }

    double maximumDifference = forwardBackardTransformationResults.maxCoeff( );
    double minimumDifference = forwardBackardTransformationResults.minCoeff( );

    BOOST_CHECK_SMALL( maximumDifference, 1.0E-9 );
    BOOST_CHECK_SMALL( std::fabs( minimumDifference ), 1.0E-9 );

    counter = 0;
    convertedValue = 0.0;
    forwardBackardTransformationResults.setZero( );
    for( std::map< double, double >::iterator differenceIterator = directInverseTimeDifferences.begin( ); differenceIterator !=
         directInverseTimeDifferences.end( ); differenceIterator++ )
    {
        convertedValue = indirectInverseDifferenceFunction( differenceIterator->first );
        forwardBackardTransformationResults( counter ) = convertedValue-differenceIterator->second;
        counter++;
    }

    maximumDifference = forwardBackardTransformationResults.maxCoeff( );
    minimumDifference = forwardBackardTransformationResults.minCoeff( );

    BOOST_CHECK_SMALL( maximumDifference, 1.0E-9 );
    BOOST_CHECK_SMALL( std::fabs( minimumDifference ), 1.0E-9 );

    std::vector< double > evaluationTimes = utilities::createVectorFromMapKeys( directTimeDifferences );

    std::function< double( const double ) > differenceFunction = earthTimeScaleConverter->getTimeDifferenceFunction(
                body_centered_coordinate_time_scale, local_proper_time_scale, "Graz" );
    std::function< double( const double ) > inverseDifferenceFunction = earthTimeScaleConverter->getTimeDifferenceFunction(
                local_proper_time_scale, body_centered_coordinate_time_scale, "Graz" );

    double convertedTime;
    forwardBackardTransformationResults.setZero( );
    for( unsigned int i = 0; i < evaluationTimes.size( ); i++ )
    {
        convertedTime = evaluationTimes[ i ] + differenceFunction( evaluationTimes[ i ] );
        forwardBackardTransformationResults( i ) = evaluationTimes[ i ] - ( convertedTime + inverseDifferenceFunction( convertedTime ) );
    }

    maximumDifference = forwardBackardTransformationResults.maxCoeff( );
    minimumDifference = forwardBackardTransformationResults.minCoeff( );

    BOOST_CHECK_SMALL( maximumDifference, 1.0E-9 );
    BOOST_CHECK_SMALL( std::fabs( minimumDifference ), 1.0E-9 );

    differenceFunction = earthTimeScaleConverter->getTimeDifferenceFunction(
                barycentric_coordinate_time_scale, local_proper_time_scale, "Graz" );
    inverseDifferenceFunction = earthTimeScaleConverter->getTimeDifferenceFunction(
                local_proper_time_scale, barycentric_coordinate_time_scale, "Graz" );


    forwardBackardTransformationResults.setZero( );
    for( unsigned int i = 0; i < evaluationTimes.size( ); i++ )
    {
        convertedTime = evaluationTimes[ i ] + differenceFunction( evaluationTimes[ i ] );
        forwardBackardTransformationResults( i ) = evaluationTimes[ i ] - ( convertedTime + inverseDifferenceFunction( convertedTime ) );
    }

    maximumDifference = forwardBackardTransformationResults.maxCoeff( );
    minimumDifference = forwardBackardTransformationResults.minCoeff( );

    BOOST_CHECK_SMALL( maximumDifference, 1.0E-12 );
    BOOST_CHECK_SMALL( std::fabs( minimumDifference ), 1.0E-12 );
}

BOOST_AUTO_TEST_CASE( ISS_proper_time_rate_iau )
{
    spice_interface::loadStandardSpiceKernels( );
        const std::string issCsvPath = "/Users/michael.plumaris/aces_data_analysis/Data/Relativistic/iss_tabulated.csv";
        Eigen::MatrixXd issData = input_output::readMatrixFromFile( issCsvPath, ",", "#" );
        const double initialEpoch = issData( 0, 0 );
        const double finalEpoch   = issData( issData.rows( ) - 1, 0 );

    const double outputTimeStep = 10.0;
    const double ephemerisBuffer = physical_constants::JULIAN_DAY;

    std::vector< std::string > bodiesToCreate{ "Sun", "Earth", "Moon" };
    const auto globalFrameOrigin      = "Earth";
    const auto globalFrameOrientation = "J2000";
    auto bodySettings = getDefaultBodySettings(
                bodiesToCreate, initialEpoch - ephemerisBuffer, finalEpoch + ephemerisBuffer, globalFrameOrigin, globalFrameOrientation );

    // WGS84 Earth shape and high-accuracy GCRS->ITRS rotation
    const double flattening       = 1.0 / 298.257223563;
    const double equatorialRadius = 6378137.0;
    bodySettings.at( "Earth" )->shapeModelSettings =
            std::make_shared< simulation_setup::OblateSphericalBodyShapeSettings >( equatorialRadius, flattening );
    bodySettings.at( "Earth" )->rotationModelSettings =
            std::make_shared< simulation_setup::GcrsToItrsRotationModelSettings >( basic_astrodynamics::iau_2006, globalFrameOrientation );

    // Promote Earth gravity to degree/order 300 if available
    auto earthGravitySettings = std::dynamic_pointer_cast< simulation_setup::SphericalHarmonicsGravityFieldSettings >(
                bodySettings.at( "Earth" )->gravityFieldSettings );
    if( earthGravitySettings != nullptr )
    {
        const int targetDegree = 300;
        const int targetOrder  = 300;
        const int currentDegree = static_cast< int >( earthGravitySettings->getCosineCoefficients( ).rows( ) ) - 1;
        const int currentOrder  = static_cast< int >( earthGravitySettings->getCosineCoefficients( ).cols( ) ) - 1;
        const int degreeToCopy  = std::min( currentDegree, targetDegree );
        const int orderToCopy   = std::min( currentOrder, targetOrder );
        Eigen::MatrixXd cosine  = Eigen::MatrixXd::Zero( targetDegree + 1, targetOrder + 1 );
        Eigen::MatrixXd sine    = Eigen::MatrixXd::Zero( targetDegree + 1, targetOrder + 1 );
        cosine.block( 0, 0, degreeToCopy + 1, orderToCopy + 1 ) =
                earthGravitySettings->getCosineCoefficients( ).block( 0, 0, degreeToCopy + 1, orderToCopy + 1 );
        sine.block( 0, 0, degreeToCopy + 1, orderToCopy + 1 ) =
                earthGravitySettings->getSineCoefficients( ).block( 0, 0, degreeToCopy + 1, orderToCopy + 1 );
        earthGravitySettings = std::make_shared< simulation_setup::SphericalHarmonicsGravityFieldSettings >(
                earthGravitySettings->getGravitationalParameter( ),
                earthGravitySettings->getReferenceRadius( ),
                cosine, sine,
                earthGravitySettings->getAssociatedReferenceFrame( ) );
        earthGravitySettings->resetAssociatedReferenceFrame( "ITRS" );
        bodySettings.at( "Earth" )->gravityFieldSettings = earthGravitySettings;
    }

    SystemOfBodies bodies = createSystemOfBodies( bodySettings );

    // ISS tabulated ephemeris (epoch [s TDB], x y z vx vy vz in meters and m/s)
    std::map< double, Eigen::Vector6d > issStateHistory;
    for( int i = 0; i < issData.rows( ); ++i )
    {
        if( issData.cols( ) >= 7 )
        {
            Eigen::Vector6d state;
            state << issData( i, 1 ), issData( i, 2 ), issData( i, 3 ),
                     issData( i, 4 ), issData( i, 5 ), issData( i, 6 );
            issStateHistory[ issData( i, 0 ) ] = state;
        }
    }
    auto issInterpolator =
            std::make_shared< interpolators::LagrangeInterpolator< double, Eigen::Vector6d > >( issStateHistory, 6 );
    auto issEphemeris = std::make_shared< ephemerides::TabulatedCartesianEphemeris< double, double > >(
            issInterpolator, globalFrameOrigin, globalFrameOrientation );
    bodies.createEmptyBody( "ISS" );
    bodies.getBody( "ISS" )->setEphemeris( issEphemeris );
    setGlobalFrameBodyEphemerides( bodies.getMap( ), globalFrameOrigin, globalFrameOrientation );

    // Initialize translational and rotational states at initial epoch
    for( const auto& bodyName : bodiesToCreate )
    {
        if( bodies.doesBodyExist( bodyName ) )
        {
            bodies.getBody( bodyName )->setStateFromEphemeris( initialEpoch );
        }
    }
    bodies.getBody( "Earth" )->setCurrentRotationalStateToLocalFrameFromEphemeris( initialEpoch );
    bodies.getBody( "ISS" )->setStateFromEphemeris( initialEpoch );
    {
        auto earthGravity = std::dynamic_pointer_cast< gravitation::SphericalHarmonicsGravityField >(
                bodies.getBody( "Earth" )->getGravityFieldModel( ) );
        auto earthRotation = bodies.getBody( "Earth" )->getRotationalEphemeris( );
        if( earthGravity != nullptr && earthRotation != nullptr )
        {
            // Use rotation at initial epoch (stable for tests; time-varying rotation is handled by the ephemeris)
            earthGravity->setRotationWrapper(
                std::make_shared< reference_frames::QuaternionRotationWrapper >(
                    [earthRotation, initialEpoch]( )
                    {
                        return earthRotation->getRotationToTargetFrame( initialEpoch );
                    } ) );
        }
    }

    // Ground stations (height [m], latitude/longitude [deg]) - others commented for now
    std::map< std::string, Eigen::Vector3d > stationGeodetic;
    stationGeodetic[ "GT101" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 48.836 ),
                                 unit_conversions::convertDegreesToRadians( 2.3344 ), 137.5458 ).finished( );
    // stationGeodetic[ "GT003" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 48.8359 ),
    //                              unit_conversions::convertDegreesToRadians( 2.3343 ), 137.5739 ).finished( );
    // stationGeodetic[ "GT007" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 51.42437417 ),
    //                              unit_conversions::convertDegreesToRadians( -0.338699387 ), 28.579 ).finished( );
    // stationGeodetic[ "GT002" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 34.20173389 ),
    //                              unit_conversions::convertDegreesToRadians( -118.1765639 ), 350.265453 ).finished( );
    // stationGeodetic[ "GT004" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 52.29646039 ),
    //                              unit_conversions::convertDegreesToRadians( 10.46394649 ), 146.399 ).finished( );
    // stationGeodetic[ "GT005" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 35.70784816 ),
    //                              unit_conversions::convertDegreesToRadians( 139.4878365 ), 135.5472 ).finished( );
    for( const auto& station : stationGeodetic )
    {
        createGroundStation(
                bodies.getBody( "Earth" ), station.first, station.second, coordinate_conversions::geodetic_position );
    }

    const double integratorStep = 10.0;
    auto integratorSettings = numerical_integrators::rungeKutta4Settings( integratorStep );
    auto terminationSettings = std::make_shared< PropagationTimeTerminationSettings >( finalEpoch );

    // Chained: Earth second-order + GT101 topocentric, ISS second-order bodycentric
    const std::vector< std::string > topocentricPerturbingBodies{ "Sun", "Moon" };
    const Eigen::Matrix< double, Eigen::Dynamic, 1 > initialRelativisticState =
            Eigen::Matrix< double, Eigen::Dynamic, 1 >::Zero( 1 );
    std::vector< std::shared_ptr< RelativisticTimeStatePropagatorSettings< double, double > > >
            bodyCentricToTopocentricConversionSettings;
    for( const auto& station : stationGeodetic )
    {
        bodyCentricToTopocentricConversionSettings.push_back(
                std::make_shared< BodycenteredToTopocentricTimePropagatorSettings< double, double > >(
                        std::make_pair( "Earth", station.first ),
                        false,
                        4,
                        true,
                        topocentricPerturbingBodies,
                        initialRelativisticState,
                        initialEpoch,
                        integratorSettings,
                        terminationSettings ) );
    }
    const std::vector< std::string > earthPerturbingBodies{ "Moon", "Sun" };
    const std::vector< std::string > issPerturbingBodies{ "Earth", "Sun", "Moon" };
    std::map< std::string, std::pair< int, int > > issSphericalHarmonics;
    issSphericalHarmonics[ "Earth" ] = std::make_pair( 300, 300 );
    std::map< std::string, std::shared_ptr< DirectRelativisticTimeConverterSettings<> > > converterSettings;
    converterSettings[ "Earth" ] = std::make_shared< DirectRelativisticTimeConverterSettings<> >(
            std::make_shared< propagators::SecondOrderBodyCenteredRelativisticTimeConverterSettings< double, double > >(
                    "Earth", earthPerturbingBodies, initialEpoch, integratorSettings, terminationSettings ),
            integratorSettings,
            bodyCentricToTopocentricConversionSettings );
    converterSettings[ "ISS" ] = std::make_shared< DirectRelativisticTimeConverterSettings<> >(
            std::make_shared< propagators::SecondOrderBodyCenteredRelativisticTimeConverterSettings< double, double > >(
                    "ISS", issPerturbingBodies, initialEpoch, integratorSettings, terminationSettings, issSphericalHarmonics ),
            integratorSettings );
    setRelativisticTimeConverters( bodies, converterSettings );

    auto earthTimeScaleConverter = bodies.getBody( "Earth" )->getTimeScaleConverter( );
    auto issTimeScaleConverter   = bodies.getBody( "ISS" )->getTimeScaleConverter( );
    BOOST_REQUIRE( earthTimeScaleConverter != nullptr );
    BOOST_REQUIRE( issTimeScaleConverter   != nullptr );

    for( const auto& station : stationGeodetic )
    {
        const std::string stationName = station.first;
        std::map< double, Eigen::VectorXd > conversionResults;
        for( double epoch = initialEpoch; epoch <= finalEpoch + std::numeric_limits< double >::epsilon( ); epoch += outputTimeStep )
        {
            Eigen::VectorXd current( 4 );
            const double tcbMinusStation = earthTimeScaleConverter->getTimeDifference(
                    barycentric_coordinate_time_scale, local_proper_time_scale, epoch, stationName );
            const double tcbMinusTcg = earthTimeScaleConverter->getTimeDifference(
                    barycentric_coordinate_time_scale, body_centered_coordinate_time_scale, epoch, stationName );
            const double tcgMinusStation = earthTimeScaleConverter->getTimeDifference(
                    body_centered_coordinate_time_scale, local_proper_time_scale, epoch, stationName );
            const double tcbMinusIss = issTimeScaleConverter->getTimeDifference(
                    barycentric_coordinate_time_scale, body_centered_coordinate_time_scale, epoch );

            current( 0 ) = tcbMinusTcg;                          // TCB - TCG
            current( 1 ) = tcgMinusStation;                      // TCG - Station
            current( 2 ) = tcbMinusIss;                          // TCB - ISS (body-centered)
            current( 3 ) = tcbMinusIss - tcbMinusStation;        // Station - ISS
            conversionResults[ epoch ] = current;
        }

        const std::string outputDirectory = "/Users/michael.plumaris/aces_data_analysis/Data/Relativistic/";
        boost::filesystem::create_directories( outputDirectory );
        input_output::writeDataMapToTextFile(
                conversionResults,
                "test_ISS_proper_time_rate_iau_" + stationName + ".dat",
                outputDirectory, "", 16 );
    }

    BOOST_CHECK_EQUAL( 1, 1 );
}

BOOST_AUTO_TEST_CASE( ISS_proper_time_rate_metric )
{
    spice_interface::loadStandardSpiceKernels( );
    // Use same ISS tabulated ephemeris and settings as IAU test
    const std::string issCsvPath = "/Users/michael.plumaris/aces_data_analysis/Data/Relativistic/iss_tabulated.csv";
    Eigen::MatrixXd issData = input_output::readMatrixFromFile( issCsvPath, ",", "#" );
    const double initialEpoch = issData( 0, 0 );
    const double finalEpoch   = issData( issData.rows( ) - 1, 0 );
    const double outputTimeStep = 10.0;
    const double ephemerisBuffer = physical_constants::JULIAN_DAY;
    std::vector< std::string > bodiesToCreate{ "Sun", "Earth", "Moon" };
    const auto globalFrameOrigin      = "Earth";
    const auto globalFrameOrientation = "J2000";
    auto bodySettings = getDefaultBodySettings(
            bodiesToCreate, initialEpoch - ephemerisBuffer, finalEpoch + ephemerisBuffer, globalFrameOrigin, globalFrameOrientation );
    // WGS84 Earth shape and high-accuracy GCRS->ITRS rotation
    const double flattening       = 1.0 / 298.257223563;
    const double equatorialRadius = 6378137.0;
    bodySettings.at( "Earth" )->shapeModelSettings =
            std::make_shared< simulation_setup::OblateSphericalBodyShapeSettings >( equatorialRadius, flattening );
    bodySettings.at( "Earth" )->rotationModelSettings =
            std::make_shared< simulation_setup::GcrsToItrsRotationModelSettings >( basic_astrodynamics::iau_2006, globalFrameOrientation );
    // Promote Earth gravity to degree/order 300 if available
    auto earthGravitySettings = std::dynamic_pointer_cast< simulation_setup::SphericalHarmonicsGravityFieldSettings >(
            bodySettings.at( "Earth" )->gravityFieldSettings );
    if( earthGravitySettings != nullptr )
    {
        const int targetDegree = 300;
        const int targetOrder  = 300;
        const int currentDegree = static_cast< int >( earthGravitySettings->getCosineCoefficients( ).rows( ) ) - 1;
        const int currentOrder  = static_cast< int >( earthGravitySettings->getCosineCoefficients( ).cols( ) ) - 1;
        const int degreeToCopy  = std::min( currentDegree, targetDegree );
        const int orderToCopy   = std::min( currentOrder, targetOrder );
        Eigen::MatrixXd cosine  = Eigen::MatrixXd::Zero( targetDegree + 1, targetOrder + 1 );
        Eigen::MatrixXd sine    = Eigen::MatrixXd::Zero( targetDegree + 1, targetOrder + 1 );
        cosine.block( 0, 0, degreeToCopy + 1, orderToCopy + 1 ) =
                earthGravitySettings->getCosineCoefficients( ).block( 0, 0, degreeToCopy + 1, orderToCopy + 1 );
        sine.block( 0, 0, degreeToCopy + 1, orderToCopy + 1 ) =
                earthGravitySettings->getSineCoefficients( ).block( 0, 0, degreeToCopy + 1, orderToCopy + 1 );
        earthGravitySettings = std::make_shared< simulation_setup::SphericalHarmonicsGravityFieldSettings >(
                earthGravitySettings->getGravitationalParameter( ),
                earthGravitySettings->getReferenceRadius( ),
                cosine, sine,
                earthGravitySettings->getAssociatedReferenceFrame( ) );
        earthGravitySettings->resetAssociatedReferenceFrame( "ITRS" );
        bodySettings.at( "Earth" )->gravityFieldSettings = earthGravitySettings;
    }
    SystemOfBodies bodies = createSystemOfBodies( bodySettings );
    // ISS tabulated ephemeris
    std::map< double, Eigen::Vector6d > issStateHistory;
    for( int i = 0; i < issData.rows( ); ++i )
    {
        if( issData.cols( ) >= 7 )
        {
            Eigen::Vector6d state;
            state << issData( i, 1 ), issData( i, 2 ), issData( i, 3 ),
                     issData( i, 4 ), issData( i, 5 ), issData( i, 6 );
            issStateHistory[ issData( i, 0 ) ] = state;
        }
    }
    auto issInterpolator =
            std::make_shared< interpolators::LagrangeInterpolator< double, Eigen::Vector6d > >( issStateHistory, 6 );
    auto issEphemeris = std::make_shared< ephemerides::TabulatedCartesianEphemeris< double, double > >(
            issInterpolator, globalFrameOrigin, globalFrameOrientation );
    bodies.createEmptyBody( "ISS" );
    bodies.getBody( "ISS" )->setEphemeris( issEphemeris );
    setGlobalFrameBodyEphemerides( bodies.getMap( ), globalFrameOrigin, globalFrameOrientation );

    for( const auto& bodyName : bodiesToCreate )
    {
        if( bodies.doesBodyExist( bodyName ) )
        {
            bodies.getBody( bodyName )->setStateFromEphemeris( initialEpoch );
        }
    }
    bodies.getBody( "Earth" )->setCurrentRotationalStateToLocalFrameFromEphemeris( initialEpoch );

    // Attach rotation wrapper for harmonic potential evaluation
    {
        auto earthGravity = std::dynamic_pointer_cast< gravitation::SphericalHarmonicsGravityField >(
                bodies.getBody( "Earth" )->getGravityFieldModel( ) );
        auto earthRotation = bodies.getBody( "Earth" )->getRotationalEphemeris( );
        if( earthGravity != nullptr && earthRotation != nullptr )
        {
            // Use rotation at initial epoch (stable for tests; time-varying rotation is handled by the ephemeris)
            earthGravity->setRotationWrapper(
                std::make_shared< reference_frames::QuaternionRotationWrapper >(
                    [earthRotation, initialEpoch]( )
                    {
                        return earthRotation->getRotationToTargetFrame( initialEpoch );
                    } ) );
        }
    }

    // Ground stations (height [m], latitude/longitude [deg]) - others commented for now
    std::map< std::string, Eigen::Vector3d > stationGeodetic;
    stationGeodetic[ "GT101" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 48.836 ),
                                 unit_conversions::convertDegreesToRadians( 2.3344 ), 137.5458 ).finished( );
    // stationGeodetic[ "GT003" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 48.8359 ),
    //                              unit_conversions::convertDegreesToRadians( 2.3343 ), 137.5739 ).finished( );
    // stationGeodetic[ "GT007" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 51.42437417 ),
    //                              unit_conversions::convertDegreesToRadians( -0.338699387 ), 28.579 ).finished( );
    // stationGeodetic[ "GT002" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 34.20173389 ),
    //                              unit_conversions::convertDegreesToRadians( -118.1765639 ), 350.265453 ).finished( );
    // stationGeodetic[ "GT004" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 52.29646039 ),
    //                              unit_conversions::convertDegreesToRadians( 10.46394649 ), 146.399 ).finished( );
    // stationGeodetic[ "GT005" ] = ( Eigen::Vector3d( ) << unit_conversions::convertDegreesToRadians( 35.70784816 ),
    //                              unit_conversions::convertDegreesToRadians( 139.4878365 ), 135.5472 ).finished( );
    for( const auto& station : stationGeodetic )
    {
        createGroundStation(
                bodies.getBody( "Earth" ), station.first, station.second, coordinate_conversions::geodetic_position );
    }

    // Metric setup
    std::vector< std::string > metricBodies{ "Sun", "Earth", "Moon" };
    auto metricSettings = std::make_shared< SolarSystemSpaceTimeMetricSettings >(
            metricBodies,
            std::vector< std::string >( ),
            std::map< std::string, std::pair< int, int > >( ),
            std::vector< std::string >( ),
            std::make_shared< relativity::PPNParameterSet >( 1.0, 1.0 ) );
    baseMetric = createSpaceTimeMetric( metricSettings, bodies );
    evaluatedMetricObjects.clear( );

    const double integratorStep = 10.0;
    auto integratorSettings = numerical_integrators::rungeKutta4Settings( integratorStep );
    auto terminationSettings = std::make_shared< PropagationTimeTerminationSettings >( finalEpoch );

    auto directOutputSettings = std::make_shared< SingleArcPropagatorProcessingSettings >(
            true, true, 1, TUDAT_NAN,
            std::make_shared< PropagationPrintSettings >( false, false ) );

    // Reset states before metric propagations
    for( const auto& bodyName : bodiesToCreate )
    {
        if( bodies.doesBodyExist( bodyName ) )
        {
            bodies.getBody( bodyName )->setStateFromEphemeris( initialEpoch );
        }
    }
    bodies.getBody( "Earth" )->setCurrentRotationalStateToLocalFrameFromEphemeris( initialEpoch );
    bodies.getBody( "ISS" )->setStateFromEphemeris( initialEpoch );

    // Direct propagation: ISS (proper time from metric)
    auto issDirectSettings =
            std::make_shared< DirectRelativisticTimePropagatorSettings< double, double > >(
                    std::make_pair( "ISS", "" ),
                    initialEpoch,
                    integratorSettings,
                    terminationSettings,
                    &basic_astrodynamics::doDummyTimeConversion< double >,
                    1.0,
                    std::vector< std::shared_ptr< SingleDependentVariableSaveSettings > >( ),
                    directOutputSettings );
    SingleArcDynamicsSimulator< double > issDirectDynamics( bodies, issDirectSettings, true );

    auto issTimeScaleConverter   = bodies.getBody( "ISS" )->getTimeScaleConverter( );
    BOOST_REQUIRE( issTimeScaleConverter   != nullptr );

    const std::string outputDirectory = "/Users/michael.plumaris/aces_data_analysis/Data/Relativistic/";
    boost::filesystem::create_directories( outputDirectory );
    for( const auto& station : stationGeodetic )
    {
        const std::string stationName = station.first;
        // Reset states before each station propagation
        for( const auto& bodyName : bodiesToCreate )
        {
            if( bodies.doesBodyExist( bodyName ) )
            {
                bodies.getBody( bodyName )->setStateFromEphemeris( initialEpoch );
            }
        }
        bodies.getBody( "Earth" )->setCurrentRotationalStateToLocalFrameFromEphemeris( initialEpoch );
        bodies.getBody( "ISS" )->setStateFromEphemeris( initialEpoch );
        // Direct propagation for this station
        auto stationDirectSettings =
                std::make_shared< DirectRelativisticTimePropagatorSettings< double, double > >(
                        std::make_pair( "Earth", stationName ),
                        initialEpoch,
                        integratorSettings,
                        terminationSettings,
                        &basic_astrodynamics::doDummyTimeConversion< double >,
                        1.0,
                        std::vector< std::shared_ptr< SingleDependentVariableSaveSettings > >( ),
                        directOutputSettings );
        SingleArcDynamicsSimulator< double > stationDirectDynamics( bodies, stationDirectSettings, true );

        auto earthTimeScaleConverter = bodies.getBody( "Earth" )->getTimeScaleConverter( );
        BOOST_REQUIRE( earthTimeScaleConverter != nullptr );

        std::map< double, Eigen::VectorXd > conversionResults;
        for( double epoch = initialEpoch; epoch <= finalEpoch + std::numeric_limits< double >::epsilon( ); epoch += outputTimeStep )
        {
            Eigen::VectorXd current( 3 );
            const double tcbMinusStation = earthTimeScaleConverter->getTimeDifference(
                    barycentric_coordinate_time_scale, local_proper_time_scale, epoch, stationName );
            const double tcbMinusIss = issTimeScaleConverter->getTimeDifference(
                    barycentric_coordinate_time_scale, local_proper_time_scale, epoch );
            current( 0 ) = tcbMinusStation;                      // TCB - Station (metric)
            current( 1 ) = tcbMinusIss;                          // TCB - ISS (metric)
            current( 2 ) = tcbMinusIss - tcbMinusStation;        // Station - ISS (metric)
            conversionResults[ epoch ] = current;
        }
        input_output::writeDataMapToTextFile(
                conversionResults,
                "test_ISS_proper_time_rate_metric_" + stationName + ".dat",
                outputDirectory, "", 16 );
    }
    BOOST_CHECK_EQUAL( 1, 1 );
}


BOOST_AUTO_TEST_SUITE_END( )

}  // namespace unit_tests

}  // namespace tudat
