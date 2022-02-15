/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 *
 *    References
 *      Wakker, K. F. (2007), Lecture Notes astro II (Chapter 18), TU Delft course AE4-874,
 *          Delft University of technology, Delft, The Netherlands.
 *
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#include "tudat/astro/basic_astro/physicalConstants.h"
#include "tudat/math/basic/mathematicalConstants.h"
#include "tudat/math/basic/basicMathematicsFunctions.h"
#include "tudat/math/basic/coordinateConversions.h"
#include "tudat/astro/ephemerides/approximatePlanetPositions.h"
#include "tudat/simulation/simulation.h"
#include "tudat/simulation/propagation_setup/propagationLowThrustProblem.h"
#include "tudat/astro/basic_astro/celestialBodyConstants.h"
#include "tudat/astro/low_thrust/shape_based/baseFunctionsSphericalShaping.h"
#include "tudat/astro/low_thrust/shape_based/compositeFunctionSphericalShaping.h"
#include "tudat/astro/low_thrust/shape_based/sphericalShaping.h"
#include "tudat/astro/low_thrust/shape_based/sphericalShapingLeg.h"

namespace tudat
{
namespace unit_tests
{

using namespace tudat::simulation_setup;
using namespace tudat::ephemerides;
using namespace tudat::shape_based_methods;
using namespace tudat::root_finders;
using namespace tudat::propagators;

//! Test spherical shaping implementation.
BOOST_AUTO_TEST_SUITE( test_spherical_shaping )

//! Test.
BOOST_AUTO_TEST_CASE( test_spherical_shaping_earth_mars_transfer )
{
    spice_interface::loadStandardSpiceKernels( );

    int numberOfRevolutions = 1;
    double julianDate = 8174.5 * physical_constants::JULIAN_DAY;
    double timeOfFlight = 580.0 * physical_constants::JULIAN_DAY;

    // Ephemeris departure body.
    EphemerisPointer pointerToDepartureBodyEphemeris = std::make_shared< ApproximateJplEphemeris>( "Earth" );
    EphemerisPointer pointerToArrivalBodyEphemeris = std::make_shared< ApproximateJplEphemeris >( "Mars" );
    Eigen::Vector6d initialState = pointerToDepartureBodyEphemeris->getCartesianState( julianDate );
    Eigen::Vector6d finalState = pointerToArrivalBodyEphemeris->getCartesianState(
                julianDate + timeOfFlight );

    // Define root finder settings (used to update the updated value of the free coefficient, so that it
    // matches the required time of flight).
    std::shared_ptr< RootFinderSettings > rootFinderSettings =
            tudat::root_finders::bisectionRootFinderSettings( 1.0E-6, TUDAT_NAN, TUDAT_NAN, 30 );

    std::cout<<"************************ ORIGINAL CLASS **************************"<<std::endl;
    // Compute shaped trajectory.
    SphericalShaping sphericalShaping = SphericalShaping(
                initialState, finalState, timeOfFlight,
                spice_interface::getBodyGravitationalParameter( "Sun" ),
                numberOfRevolutions, 0.000703, rootFinderSettings, 1.0e-6, 1.0e-1 );

    std::cout<<"************************ NEW CLASS **************************"<<std::endl;
    SphericalShapingLeg sphericalShapingLeg = SphericalShapingLeg(
            pointerToDepartureBodyEphemeris, pointerToArrivalBodyEphemeris,
            spice_interface::getBodyGravitationalParameter("Sun"),
            numberOfRevolutions, rootFinderSettings, 0.000703, 1.0e-6, 1.0e-1);
    sphericalShapingLeg.updateLegParameters( (
            Eigen::Vector2d( )<< julianDate, julianDate + timeOfFlight ).finished( ) );

    // Initialise peak acceleration.
    double peakThrustAcceleration_old = 0.0;

    double stepSize_old = (sphericalShaping.getFinalValueInpendentVariable( ) -
                           sphericalShaping.getInitialValueInpendentVariable( ) ) / 5000.0;
    for ( int i = 0 ; i <= 5000 ; i++ )
    {
        double currentThetaAngle = sphericalShaping.getInitialValueInpendentVariable() + i * stepSize_old;
        if (sphericalShaping.computeCurrentThrustAcceleration( currentThetaAngle ).norm() > peakThrustAcceleration_old )
        {
            peakThrustAcceleration_old = sphericalShaping.computeCurrentThrustAcceleration(currentThetaAngle ).norm();
        }
    }

    // Initialise peak acceleration
    double peakThrustAcceleration = 0.0;

    // Loop over independent variable values and check peak acceleration
    double stepSize = (sphericalShapingLeg.getFinalValueAzimuth() -
                       sphericalShapingLeg.getInitialValueAzimuth() ) / 5000.0;
    for ( int i = 0 ; i <= 5000 ; i++ )
    {
        double currentThetaAngle = sphericalShapingLeg.getInitialValueAzimuth() + i * stepSize;
        if (sphericalShapingLeg.computeCurrentThrustAccelerationFromAzimuth(currentThetaAngle).norm() > peakThrustAcceleration )
        {
            peakThrustAcceleration = sphericalShapingLeg.computeCurrentThrustAccelerationFromAzimuth(currentThetaAngle).norm();
        }
    }


    // Check results consistency w.r.t. Roegiers, T., Application of the Spherical Shaping Method to a Low-Thrust
    // Multiple Asteroid Rendezvous Mission, TU Delft (MSc thesis), 2014
    double expectedDeltaV = 5700.0;
    double expectedPeakAcceleration = 2.4e-4;

    std::cout<<sphericalShaping.computeDeltaV()<<" "<<sphericalShapingLeg.getLegDeltaV()<<std::endl;
    std::cout<<peakThrustAcceleration_old<<" "<<peakThrustAcceleration<<std::endl;
    // DeltaV provided with a precision of 5 m/s
    BOOST_CHECK_SMALL( std::fabs(  sphericalShaping.computeDeltaV() - expectedDeltaV ), 5.0 );
    BOOST_CHECK_SMALL( std::fabs(  sphericalShapingLeg.getLegDeltaV() - expectedDeltaV ), 5.0 );
    // Peak acceleration provided with a precision 2.0e-6 m/s^2
    BOOST_CHECK_SMALL(std::fabs(peakThrustAcceleration_old - expectedPeakAcceleration ), 2.0e-6 );
    BOOST_CHECK_SMALL(std::fabs(peakThrustAcceleration - expectedPeakAcceleration ), 2.0e-6 );

}


//! Test.
BOOST_AUTO_TEST_CASE( test_spherical_shaping_earth_1989ML_transfer )
{
    spice_interface::loadStandardSpiceKernels( );

    int numberOfRevolutions = 1;
    double julianDate = 7799.5 * physical_constants::JULIAN_DAY;
    double timeOfFlight = 600.0 * physical_constants::JULIAN_DAY;

    // Ephemeris departure body.
    EphemerisPointer pointerToDepartureBodyEphemeris = std::make_shared< ApproximateJplEphemeris>( "Earth" );
    Eigen::Vector6d initialState = pointerToDepartureBodyEphemeris->getCartesianState( julianDate );

    // Final state derived from ML1989 ephemeris (from Spice).
    Eigen::Vector6d finalState = (
                Eigen::Vector6d() <<
                1.197701029846094E+00 * physical_constants::ASTRONOMICAL_UNIT,
                1.653518856610793E-01 * physical_constants::ASTRONOMICAL_UNIT,
                - 9.230177854743750E-02 * physical_constants::ASTRONOMICAL_UNIT,
                - 4.891080912584867E-05 * physical_constants::ASTRONOMICAL_UNIT / physical_constants::JULIAN_DAY,
                1.588950249593135E-02 * physical_constants::ASTRONOMICAL_UNIT / physical_constants::JULIAN_DAY,
                - 2.980245580772588E-04 * physical_constants::ASTRONOMICAL_UNIT / physical_constants::JULIAN_DAY ).finished();

    // Define root finder settings (used to update the updated value of the free coefficient, so that it matches the required time of flight).
    std::shared_ptr< RootFinderSettings > rootFinderSettings =
            tudat::root_finders::bisectionRootFinderSettings( 1.0E-6, TUDAT_NAN, TUDAT_NAN, 30 );

    // Compute shaped trajectory.
    SphericalShaping sphericalShaping = SphericalShaping(
                initialState, finalState, timeOfFlight,
                spice_interface::getBodyGravitationalParameter( "Sun" ),
                numberOfRevolutions, -0.0000703, rootFinderSettings, -1.0e-2, 1.0e-2 );


    // Compute step size.
    double numberOfSteps = 5000.0;
    double stepSize = ( sphericalShaping.getFinalValueInpendentVariable( ) -
                        sphericalShaping.getInitialValueInpendentVariable( ) ) / numberOfSteps;

    // Initialise peak acceleration.
    double peakThrustAcceleration = 0.0;

    // Compute peak acceleration.
    for ( int i = 0 ; i <= numberOfSteps ; i++ )
    {
        double currentThetaAngle = sphericalShaping.getInitialValueInpendentVariable() + i * stepSize;

        if ( sphericalShaping.computeCurrentThrustAcceleration( currentThetaAngle ).norm() >  peakThrustAcceleration )
        {
            peakThrustAcceleration = sphericalShaping.computeCurrentThrustAcceleration( currentThetaAngle ).norm();
        }
    }


    // Check results consistency w.r.t. Roegiers, T., Application of the Spherical Shaping Method to a Low-Thrust
    // Multiple Asteroid Rendezvous Mission, TU Delft (MSc thesis), 2014
    // The expected differences are a bit larger than for Earth-Mars transfer due to the higher uncertainty in 1989ML's ephemeris.
    double expectedDeltaV = 4530.0;
    double expectedPeakAcceleration = 1.8e-4;

    // DeltaV provided with a precision of 0.1 km/s
    BOOST_CHECK_SMALL( std::fabs(  sphericalShaping.computeDeltaV() - expectedDeltaV ), 100.0 );
    // Peak acceleration provided with a precision 1.0e-5 m/s^2
    BOOST_CHECK_SMALL( std::fabs(  peakThrustAcceleration - expectedPeakAcceleration ), 1e-5 );

}


//! Test spherical shaping method with various number of revolutions.
BOOST_AUTO_TEST_CASE( test_spherical_shaping_earth_mars_transfer_multi_revolutions )
{
    spice_interface::loadStandardSpiceKernels( );

    double julianDate = 8174.5 * physical_constants::JULIAN_DAY;

    std::vector< int > numberOfRevolutionsVector = { 0, 1, 2 };
    std::vector< double > timeOfFlightVector = { 300.0, 580.0, 750.0 };

    // Ephemeris departure body.
    EphemerisPointer pointerToDepartureBodyEphemeris = std::make_shared< ApproximateJplEphemeris>(
                "Earth"  );
    EphemerisPointer pointerToArrivalBodyEphemeris = std::make_shared< ApproximateJplEphemeris >(
                "Mars"  );

    // Define root finder settings (used to update the updated value of the free coefficient, so that it matches the required time of flight).
    std::shared_ptr< RootFinderSettings > rootFinderSettings =
            tudat::root_finders::bisectionRootFinderSettings( 1.0E-6, TUDAT_NAN, TUDAT_NAN, 30 );

    // Bounds for the free parameter.
    std::vector< double > freeParameterLowerBoundVector = { -1.0, 1.0e-6, -1.0e-2 };
    std::vector< double > freeParameterUpperBoundVector = { 5.0e-1, 1.0e-1, 1.0e-2 };

    // Define initial state.
    Eigen::Vector6d initialState = pointerToDepartureBodyEphemeris->getCartesianState( julianDate );

    for ( unsigned int currentTestCase = 0 ; currentTestCase < numberOfRevolutionsVector.size( ) ; currentTestCase++ )
    {
        // Define final state.
        Eigen::Vector6d finalState = pointerToArrivalBodyEphemeris->getCartesianState(
                    julianDate + timeOfFlightVector.at( currentTestCase )* physical_constants::JULIAN_DAY );

        // Compute shaped trajectory.
        SphericalShaping sphericalShaping = SphericalShaping(
                    initialState, finalState, timeOfFlightVector.at( currentTestCase )* physical_constants::JULIAN_DAY,
                    spice_interface::getBodyGravitationalParameter( "Sun" ),
                    numberOfRevolutionsVector.at( currentTestCase ),
                    0.000703, rootFinderSettings, freeParameterLowerBoundVector.at( currentTestCase ),
                    freeParameterUpperBoundVector.at( currentTestCase ) );

        SphericalShapingLeg sphericalShapingLeg = SphericalShapingLeg(
                pointerToDepartureBodyEphemeris, pointerToArrivalBodyEphemeris,
                spice_interface::getBodyGravitationalParameter("Sun"),
                numberOfRevolutionsVector.at(currentTestCase), rootFinderSettings, 0.000703,
                freeParameterLowerBoundVector.at(currentTestCase), freeParameterUpperBoundVector.at(currentTestCase));
        sphericalShapingLeg.updateLegParameters( (
                Eigen::Vector2d( )<< julianDate, julianDate + timeOfFlightVector.at( currentTestCase ) * physical_constants::JULIAN_DAY ).finished( ) );

        // Check consistency of final azimuth angle value with required number of revolutions.
        double initialAzimuthAngle_old = sphericalShaping.getInitialValueInpendentVariable( );
        double finalAzimuthAngle_old = sphericalShaping.getFinalValueInpendentVariable( );

        double initialAzimuthAngle = sphericalShapingLeg.getInitialValueAzimuth();
        double finalAzimuthAngle = sphericalShapingLeg.getFinalValueAzimuth();

        double expectedInitialAzimuthAngle = coordinate_conversions::convertCartesianToSphericalState( initialState )[ 1 ];
        if ( expectedInitialAzimuthAngle < 0.0 )
        {
            expectedInitialAzimuthAngle += 2.0 * mathematical_constants::PI;
        }
        double expectedFinalAzimuthAngle = coordinate_conversions::convertCartesianToSphericalState( finalState )[ 1 ];
        if ( expectedFinalAzimuthAngle < 0.0 )
        {
            expectedFinalAzimuthAngle += 2.0 * mathematical_constants::PI;
        }

        if ( expectedFinalAzimuthAngle - expectedInitialAzimuthAngle < 0 )
        {
            expectedFinalAzimuthAngle += 2.0 * mathematical_constants::PI * ( numberOfRevolutionsVector.at( currentTestCase )+ 1 );
        }
        else
        {
            expectedFinalAzimuthAngle += 2.0 * mathematical_constants::PI * numberOfRevolutionsVector.at( currentTestCase );
        }

        BOOST_CHECK_SMALL(std::fabs(initialAzimuthAngle_old - expectedInitialAzimuthAngle ), 1.0e-15 );
        BOOST_CHECK_SMALL(std::fabs(finalAzimuthAngle_old - expectedFinalAzimuthAngle ), 1.0e-15 );

        BOOST_CHECK_SMALL(std::fabs(initialAzimuthAngle - expectedInitialAzimuthAngle ), 1.0e-15 );
        BOOST_CHECK_SMALL(std::fabs(finalAzimuthAngle - expectedFinalAzimuthAngle ), 1.0e-15 );

        // Check consistency of expected and calculated states (both at departure and arrival).
        for ( int i = 0 ; i < 6 ; i++ )
        {
            BOOST_CHECK_SMALL( std::fabs( ( initialState[ i ] - sphericalShaping.computeCurrentStateVector(initialAzimuthAngle_old )[ i ] )
                                          / initialState[ i ] ), 1.0e-12 );
            BOOST_CHECK_SMALL( std::fabs( ( finalState[ i ] - sphericalShaping.computeCurrentStateVector(finalAzimuthAngle_old )[ i ] )
                                          / finalState[ i ] ), 1.0e-12 );

            BOOST_CHECK_SMALL( std::fabs( ( initialState[ i ] -
                    sphericalShapingLeg.computeCurrentStateVectorFromAzimuth(initialAzimuthAngle)[ i ] )
                                          / initialState[ i ] ), 1.0e-12 );
            BOOST_CHECK_SMALL( std::fabs( ( finalState[ i ] -
                    sphericalShapingLeg.computeCurrentStateVectorFromAzimuth(finalAzimuthAngle)[ i ] )
                                          / finalState[ i ] ), 1.0e-12 );
        }
    }
}

SystemOfBodies getTestBodyMap( )
{
    // Create central, departure and arrival bodies.
    std::vector< std::string > bodiesToCreate;
    bodiesToCreate.push_back( "Sun" );
    bodiesToCreate.push_back( "Earth" );
    bodiesToCreate.push_back( "Mars" );
    bodiesToCreate.push_back( "Jupiter" );


    std::string frameOrigin = "SSB";
    std::string frameOrientation = "ECLIPJ2000";

    BodyListSettings bodySettings =
            getDefaultBodySettings( bodiesToCreate, frameOrigin, frameOrientation );

    // Define central body ephemeris settings.
    bodySettings.at( "Sun" )->ephemerisSettings = std::make_shared< ConstantEphemerisSettings >(
                ( Eigen::Vector6d( ) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ).finished( ), frameOrigin, frameOrientation );


    // Create system of bodies.
    SystemOfBodies bodies = createSystemOfBodies( bodySettings );

    bodies.createEmptyBody( "Vehicle" );

    return bodies;

}

BOOST_AUTO_TEST_CASE( test_spherical_shaping_full_propagation )
{

    int numberOfRevolutions = 1;
    double julianDate = 8174.5 * physical_constants::JULIAN_DAY;
    double  timeOfFlight = 580.0;

    // Ephemeris for arrival and departure body.
    EphemerisPointer pointerToDepartureBodyEphemeris = std::make_shared< ApproximateJplEphemeris>(
                "Earth"  );
    EphemerisPointer pointerToArrivalBodyEphemeris = std::make_shared< ApproximateJplEphemeris >(
                "Mars"  );
    Eigen::Vector6d stateAtDeparture = pointerToDepartureBodyEphemeris->getCartesianState( julianDate );
    Eigen::Vector6d stateAtArrival = pointerToArrivalBodyEphemeris->getCartesianState(
                julianDate + timeOfFlight * physical_constants::JULIAN_DAY );

    // Define root finder settings (used to update the updated value of the free coefficient, so that it matches the required time of flight).
    std::shared_ptr< RootFinderSettings > rootFinderSettings =
            tudat::root_finders::bisectionRootFinderSettings( 1.0E-6, TUDAT_NAN, TUDAT_NAN, 30 );

    // Compute shaped trajectory.
    std::shared_ptr< SphericalShaping > sphericalShaping = std::make_shared< SphericalShaping >(
                stateAtDeparture, stateAtArrival, timeOfFlight * physical_constants::JULIAN_DAY,
                spice_interface::getBodyGravitationalParameter( "Sun" ),
                numberOfRevolutions, 0.000703,
                rootFinderSettings, 1.0e-6, 1.0e-1 );

    std::map< double, Eigen::VectorXd > fullPropagationResults;
    std::map< double, Eigen::Vector6d > shapingMethodResults;
    std::map< double, Eigen::VectorXd > dependentVariablesHistory;

    // Create system of bodies
    SystemOfBodies bodies = getTestBodyMap( );
    bodies.at( "Vehicle" )->setBodyMassFunction(  [ = ]( const double currentTime ){ return 2000.0; } );


    // Define integrator settings
    std::shared_ptr< numerical_integrators::IntegratorSettings< double > > integratorSettings =
            std::make_shared< numerical_integrators::IntegratorSettings< double > > (
                numerical_integrators::rungeKutta4, 0.0, timeOfFlight * physical_constants::JULIAN_DAY / ( 1000.0 ) );

    // Define mass and specific impulse functions of the vehicle.
    std::function< double( const double ) > specificImpulseFunction =
            [ ]( const double ){ return 3000.0; };


    // Create object with list of dependent variables
    std::vector< std::shared_ptr< SingleDependentVariableSaveSettings > > dependentVariablesList;
    dependentVariablesList.push_back( std::make_shared< SingleAccelerationDependentVariableSaveSettings >(
                                          basic_astrodynamics::thrust_acceleration, "Vehicle", "Vehicle", 0 ) );
    std::shared_ptr< DependentVariableSaveSettings > dependentVariablesToSave =
            std::make_shared< DependentVariableSaveSettings >( dependentVariablesList, false );

    // Create complete propagation settings (backward and forward propagations).
    basic_astrodynamics::AccelerationMap lowThrustAccelerationsMap =
            retrieveLowThrustAccelerationMap(
                sphericalShaping, bodies, "Vehicle", "Sun", specificImpulseFunction, 0.0 );
    std::pair< std::shared_ptr< PropagatorSettings< double > >,
            std::shared_ptr< PropagatorSettings< double > > > propagatorSettings =
            createLowThrustTranslationalStatePropagatorSettings(
                sphericalShaping, "Vehicle", "Sun", lowThrustAccelerationsMap, dependentVariablesToSave );

    // Compute shaped trajectory and propagated trajectory.
    computeLowThrustLegSemiAnalyticalAndFullPropagation(
                sphericalShaping, bodies, integratorSettings, propagatorSettings,
                fullPropagationResults, shapingMethodResults, dependentVariablesHistory );

    // Check difference between full propagation and shaping method at arrival
    // (disregarding the very last values because of expected interpolation errors).
    int numberOfDisregardedValues = 7;
    std::map< double, Eigen::VectorXd >::iterator itr = fullPropagationResults.end();
    for( int i = 0 ; i < numberOfDisregardedValues ; i++ )
    {
        itr--;
    }

    // Check results consistency between full propagation and shaped trajectory at arrival.
    for ( int i = 0 ; i < 6 ; i++ )
    {
        BOOST_CHECK_SMALL( std::fabs( shapingMethodResults[ itr->first ][ i ] - itr->second[ i ] ) /
                shapingMethodResults[ itr->first ][ i ] , 1.0e-6 );
    }

    // Check difference between full propagation and shaping method at departure
    // (disregarding the very first values because of expected interpolation errors).
    itr = fullPropagationResults.begin();
    for( int i = 0 ; i < numberOfDisregardedValues ; i++ )
    {
        itr++;
    }

    // Check results consistency between full propagation and shaped trajectory at departure.
    for ( int i = 0 ; i < 6 ; i++ )
    {
        BOOST_CHECK_SMALL( std::fabs( shapingMethodResults[ itr->first ][ i ] - itr->second[ i ] ) /
                shapingMethodResults[ itr->first ][ i ] , 1.0e-6 );
    }
}


//BOOST_AUTO_TEST_CASE( test_spherical_shaping_full_propagation_mass_propagation )
//{

//    spice_interface::loadStandardSpiceKernels( );


//    int numberOfRevolutions = 1;
//    double julianDate = 8174.5 * physical_constants::JULIAN_DAY;
//    double  timeOfFlight = 580.0;
//    double initialMass = 2000.0;
//    std::function< double( const double ) > specificImpulseFunction = [ = ]( const double ) { return 3000.0; };

//    // Ephemeris for arrival and departure body.
//    EphemerisPointer pointerToDepartureBodyEphemeris = std::make_shared< ApproximateJplEphemeris>(
//                "Earth"  );
//    EphemerisPointer pointerToArrivalBodyEphemeris = std::make_shared< ApproximateJplEphemeris >(
//                "Mars"  );
//    Eigen::Vector6d stateAtDeparture = pointerToDepartureBodyEphemeris->getCartesianState( julianDate );
//    Eigen::Vector6d stateAtArrival = pointerToArrivalBodyEphemeris->getCartesianState( julianDate + timeOfFlight * physical_constants::JULIAN_DAY );

//    // Define root finder settings (used to update the updated value of the free coefficient, so that it matches the required time of flight).
//    std::shared_ptr< RootFinderSettings > rootFinderSettings =
//            std::make_shared< RootFinderSettings >( bisection_root_finder, 1.0e-6, 30 );

//    // Compute shaped trajectory.
//    SphericalShaping sphericalShaping = SphericalShaping(
//                stateAtDeparture, stateAtArrival, timeOfFlight * physical_constants::JULIAN_DAY,
//                spice_interface::getBodyGravitationalParameter( "Sun" ),
//                numberOfRevolutions, 0.000703,
//                rootFinderSettings, 1.0e-6, 1.0e-1, initialMass );

//    std::map< double, Eigen::VectorXd > fullPropagationResults;
//    std::map< double, Eigen::Vector6d > shapingMethodResults;
//    std::map< double, Eigen::VectorXd > dependentVariablesHistory;

//    // Create system of bodies
//    SystemOfBodies bodies = getTestBodyMap( );
//    bodies.at( "Vehicle" )->setConstantBodyMass( initialMass );


//    // Define integrator settings
//    std::shared_ptr< numerical_integrators::IntegratorSettings< double > > integratorSettings =
//            std::make_shared< numerical_integrators::IntegratorSettings< double > > (
//                numerical_integrators::rungeKutta4, 0.0,
//                timeOfFlight * physical_constants::JULIAN_DAY / ( 1000.0 ) );


//    // Define list of dependent variables to save.
//    std::vector< std::shared_ptr< SingleDependentVariableSaveSettings > > dependentVariablesList;
//    dependentVariablesList.push_back( std::make_shared< SingleAccelerationDependentVariableSaveSettings >(
//                                          basic_astrodynamics::thrust_acceleration, "Vehicle", "Vehicle", 0 ) );
//    dependentVariablesList.push_back( std::make_shared< SingleDependentVariableSaveSettings >(
//                                          total_mass_rate_dependent_variables, "Vehicle" ) );

//    // Create object with list of dependent variables
//    std::shared_ptr< DependentVariableSaveSettings > dependentVariablesToSave =
//            std::make_shared< DependentVariableSaveSettings >( dependentVariablesList, false );

//    // Create termination conditions settings.
//    std::pair< std::shared_ptr< PropagationTerminationSettings >, std::shared_ptr< PropagationTerminationSettings > > terminationConditions =
//            std::make_pair( std::make_shared< PropagationTimeTerminationSettings >( 0.0 ),
//                            std::make_shared< PropagationTimeTerminationSettings >( timeOfFlight * physical_constants::JULIAN_DAY ) );


//    // Create complete propagation settings (backward and forward propagations).
//    std::pair< std::shared_ptr< PropagatorSettings< double > >,
//            std::shared_ptr< PropagatorSettings< double > > > propagatorSettings = sphericalShaping.createLowThrustPropagatorSettings(
//                bodies, "Vehicle", "Sun", specificImpulseFunction, basic_astrodynamics::AccelerationMap( ), integratorSettings, dependentVariablesToSave );

//    // Compute shaped trajectory and propagated trajectory.
//    sphericalShaping.computeSemiAnalyticalAndFullPropagation(
//                bodies, integratorSettings, propagatorSettings,
//                fullPropagationResults, shapingMethodResults, dependentVariablesHistory );



//    // Check difference between full propagation and shaping method at arrival
//    // (disregarding the very last values because of expected interpolation errors).
//    int numberOfDisregardedValues = 7;
//    std::map< double, Eigen::VectorXd >::iterator itr = fullPropagationResults.end();
//    for( int i = 0 ; i < numberOfDisregardedValues ; i++ )
//    {
//        itr--;
//    }

//    // Check results consistency between full propagation and shaped trajectory at arrival.
//    for ( int i = 0 ; i < 6 ; i++ )
//    {
////        BOOST_CHECK_SMALL( std::fabs( shapingMethodResults[ itr->first ][ i ] - itr->second[ i ] ) / shapingMethodResults[ itr->first ][ i ] , 1.0e-6 );
//    }



//    // Check difference between full propagation and shaping method at departure
//    // (disregarding the very first values because of expected interpolation errors).
//    itr = fullPropagationResults.begin();
//    for( int i = 0 ; i < numberOfDisregardedValues ; i++ )
//    {
//        itr++;
//    }

//    // Check results consistency between full propagation and shaped trajectory at departure.
//    for ( int i = 0 ; i < 6 ; i++ )
//    {
//        BOOST_CHECK_SMALL( std::fabs( shapingMethodResults[ itr->first ][ i ] - itr->second[ i ] ) / shapingMethodResults[ itr->first ][ i ] , 1.0e-6 );
//    }

//    // Check consistency between current and expected mass rates.
//    for ( std::map< double, Eigen::VectorXd >::iterator itr = dependentVariablesHistory.begin() ; itr != dependentVariablesHistory.end() ; itr++ )
//    {
//        Eigen::Vector3d currentThrustAccelerationVector = itr->second.segment( 0, 3 );
//        double currentMass = fullPropagationResults.at( itr->first )( 6 );
//        double currentMassRate = - itr->second( 3 );
//        double expectedMassRate = currentThrustAccelerationVector.norm() * currentMass /
//                ( specificImpulseFunction( itr->first ) * physical_constants::SEA_LEVEL_GRAVITATIONAL_ACCELERATION );
//        BOOST_CHECK_SMALL( std::fabs( currentMassRate - expectedMassRate ), 1.0e-15 );

//    }


//    // Test trajectory function.
//    std::vector< double > epochsVector;
//    epochsVector.push_back( 0.0 );
//    epochsVector.push_back( timeOfFlight / 4.0 * physical_constants::JULIAN_DAY );
//    epochsVector.push_back( timeOfFlight / 2.0 * physical_constants::JULIAN_DAY );
//    epochsVector.push_back( 3.0 * timeOfFlight / 4.0 * physical_constants::JULIAN_DAY );
//    epochsVector.push_back( timeOfFlight * physical_constants::JULIAN_DAY );

//    std::map< double, Eigen::Vector6d > trajectory;
//    std::map< double, Eigen::VectorXd > massProfile;
//    std::map< double, Eigen::VectorXd > thrustProfile;
//    std::map< double, Eigen::VectorXd > thrustAccelerationProfile;

//    sphericalShaping.getTrajectory( epochsVector, trajectory );
//    sphericalShaping.getMassProfile( epochsVector, massProfile, specificImpulseFunction, integratorSettings );
////    sphericalShaping.getThrustForceProfile( epochsVector, thrustProfile, specificImpulseFunction, integratorSettings );
////    sphericalShaping.getThrustAccelerationProfile( epochsVector, thrustAccelerationProfile, specificImpulseFunction, integratorSettings );

//    for ( int i = 0 ; i < 3 ; i ++ )
//    {
//        BOOST_CHECK_SMALL( std::fabs( ( trajectory.begin( )->second[ i ] - stateAtDeparture[ i ] ) /
//                                      physical_constants::ASTRONOMICAL_UNIT ), 1.0e-6 );
//        BOOST_CHECK_SMALL( std::fabs( ( trajectory.begin( )->second[ i + 3 ] - stateAtDeparture[ i + 3 ] ) /
//                ( physical_constants::ASTRONOMICAL_UNIT / physical_constants::JULIAN_YEAR ) ), 1.0e-6 );
//        BOOST_CHECK_SMALL( std::fabs( ( trajectory.rbegin( )->second[ i ] - stateAtArrival[ i ] ) /
//                                      physical_constants::ASTRONOMICAL_UNIT ), 1.0e-6 );
//        BOOST_CHECK_SMALL( std::fabs( ( trajectory.rbegin( )->second[ i + 3 ] - stateAtArrival[ i + 3 ] ) /
//                ( physical_constants::ASTRONOMICAL_UNIT / physical_constants::JULIAN_YEAR ) ), 1.0e-6 );
//    }

//    for ( std::map< double, Eigen::Vector6d >::iterator itr = trajectory.begin( ) ; itr != trajectory.end( ) ; itr++ )
//    {
//        double independentVariable = sphericalShaping.convertTimeToAzimuth( itr->first );
//        Eigen::Vector6d stateVector = sphericalShaping.computeCurrentStateVectorFromAzimuth( independentVariable );
//        Eigen::Vector3d thrustAccelerationVector = sphericalShaping.computeCurrentThrustAccelerationFromAzimuth( itr->first, specificImpulseFunction, integratorSettings );
//        Eigen::Vector3d thrustVector = sphericalShaping.computeCurrentThrustForce( itr->first, specificImpulseFunction, integratorSettings );
//        double mass = sphericalShaping.computeCurrentMass( itr->first, specificImpulseFunction, integratorSettings );

//        for ( int i = 0 ; i < 3 ; i++ )
//        {
//            BOOST_CHECK_SMALL( std::fabs( itr->second[ i ] - stateVector[ i ] ), 1.0e-6 );
//            BOOST_CHECK_SMALL( std::fabs( itr->second[ i + 3 ] - stateVector[ i + 3 ] ), 1.0e-12 );
//            BOOST_CHECK_SMALL( std::fabs( thrustAccelerationProfile[ itr->first ][ i ] - thrustAccelerationVector[ i ] ), 1.0e-6 );
//            BOOST_CHECK_SMALL( std::fabs( thrustProfile[ itr->first ][ i ] - thrustVector[ i ] ), 1.0e-12 );
//        }
//        BOOST_CHECK_SMALL( std::fabs( massProfile[ itr->first ][ 0 ] - mass ), 1.0e-10 );
//    }

//}

BOOST_AUTO_TEST_SUITE_END( )

} // namespace unit_tests
} // namespace tudat
