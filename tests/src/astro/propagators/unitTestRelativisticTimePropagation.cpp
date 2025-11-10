#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

#include <Eigen/Core>

#include "tudat/basics/testMacros.h"

#include "tudat/interface/spice/spiceInterface.h"

#include "tudat/astro/relativity/metric.h"

#include "tudat/simulation/environment_setup/createBodies.h"
#include "tudat/simulation/environment_setup/createGroundStations.h"
#include "tudat/simulation/environment_setup/createMetric.h"
#include "tudat/simulation/environment_setup/createRelativisticTimeConverter.h"

#include "tudat/simulation/propagation_setup/dynamicsSimulator.h"

namespace tudat
{

namespace unit_tests
{

using namespace simulation_setup;
using namespace numerical_integrators;
using namespace propagators;
using namespace spice_interface;
using namespace basic_astrodynamics;

BOOST_AUTO_TEST_SUITE( test_relativistic_time_propagation )

BOOST_AUTO_TEST_CASE( compareDirectMetricAndMultiTypePropagation )
{
    const std::string kernelsPath = paths::getSpiceKernelPath( );
    loadStandardSpiceKernels( );
    loadSpiceKernelInTudat( kernelsPath + "/de-403-masses.tpc" );
    loadSpiceKernelInTudat( kernelsPath + "/de440.bsp" );
    loadSpiceKernelInTudat( kernelsPath + "/naif0012.tls" );

    const double initialEphemerisTime = 1.0E7;
    const double finalEphemerisTime = initialEphemerisTime + 3.0E6;
    const double timeStep = 200.0;

    std::vector< std::string > bodyNames{ "Sun", "Earth", "Moon" };
    auto bodySettings = getDefaultBodySettings( bodyNames, initialEphemerisTime - 1.0E5, finalEphemerisTime + 1.0E5 );

    std::map< std::pair< std::string, std::string >, Eigen::Vector3d > groundStations;
    groundStations[ std::make_pair( "Earth", "Graz" ) ] =
            ( Eigen::Vector3d( ) << 4194511.7, 1162789.7, 4647362.5 ).finished( );

    SystemOfBodies bodiesDirect = createSystemOfBodies( bodySettings );
    SystemOfBodies bodiesMulti = createSystemOfBodies( bodySettings );

    createGroundStations( bodiesDirect, groundStations );
    createGroundStations( bodiesMulti, groundStations );

    setGlobalFrameBodyEphemerides( bodiesDirect.getMap( ), "SSB", "ECLIPJ2000" );
    setGlobalFrameBodyEphemerides( bodiesMulti.getMap( ), "SSB", "ECLIPJ2000" );

    const std::vector< std::string > perturbingBodies{ "Sun", "Moon" };

    baseMetric = createSpaceTimeMetric(
            std::make_shared< SolarSystemSpaceTimeMetricSettings >( perturbingBodies ),
            bodiesDirect );
    evaluatedMetricObjects.clear( );
    evaluatedMetricObjects[ std::make_pair( "Earth", "Graz" ) ] = baseMetric->Clone( );

    auto integratorSettingsDirect = numerical_integrators::rungeKutta4Settings( initialEphemerisTime, timeStep );
    auto terminationSettings = std::make_shared< PropagationTimeTerminationSettings >( finalEphemerisTime );

    auto directPropagatorSettings = std::make_shared< DirectRelativisticTimePropagatorSettings< double, double > >(
            std::make_pair( "Earth", "Graz" ),
            initialEphemerisTime,
            integratorSettingsDirect,
            terminationSettings );

    SingleArcDynamicsSimulator< double > directSimulator(
            bodiesDirect, integratorSettingsDirect, directPropagatorSettings, true, false, true );

    auto integratorSettingsMulti = numerical_integrators::rungeKutta4Settings( initialEphemerisTime, timeStep );

    auto earthProperTimeSettings =
            std::make_shared< SecondOrderBodyCenteredRelativisticTimeConverterSettings< double, double > >(
                    "Earth", perturbingBodies, initialEphemerisTime, integratorSettingsMulti, terminationSettings );

    Eigen::Matrix< double, Eigen::Dynamic, 1 > initialRelativisticState =
            Eigen::Matrix< double, Eigen::Dynamic, 1 >::Zero( 1 );

    std::vector< std::shared_ptr< RelativisticTimeStatePropagatorSettings< double, double > > > bodyToTopoSettings;
    bodyToTopoSettings.push_back(
            std::make_shared< BodycenteredToTopocentricTimePropagatorSettings< double, double > >(
                    std::make_pair( "Earth", "Graz" ),
                    false,
                    0,
                    false,
                    perturbingBodies,
                    initialRelativisticState,
                    initialEphemerisTime,
                    integratorSettingsMulti,
                    terminationSettings ) );

    auto converterSettings =
            std::make_shared< DirectRelativisticTimeConverterSettings<> >(
                    earthProperTimeSettings,
                    integratorSettingsMulti,
                    bodyToTopoSettings );

    setRelativisticTimeConverter( converterSettings, bodiesMulti );

    auto directEphemeris = bodiesDirect.getBody( "Earth" )->getTimeScaleConverter( );
    auto multiEphemeris = bodiesMulti.getBody( "Earth" )->getTimeScaleConverter( );

    auto directFunction = directEphemeris->getTimeDifferenceFunction(
            barycentric_coordinate_time_scale, local_proper_time_scale, "Graz" );
    auto multiFunction = multiEphemeris->getTimeDifferenceFunction(
            barycentric_coordinate_time_scale, local_proper_time_scale, "Graz" );

    double currentTime = initialEphemerisTime + 1000.0;
    while ( currentTime < finalEphemerisTime - 1000.0 )
    {
        const double difference = directFunction( currentTime ) - multiFunction( currentTime );
        BOOST_CHECK_SMALL( difference, 5.0E-13 );
        currentTime += 6000.0;
    }
}

BOOST_AUTO_TEST_SUITE_END( )

}  // namespace unit_tests

}  // namespace tudat
