#include <cmath>

#include "tudat/basics/utilities.h"
#include "tudat/astro/propagators/integrateEquations.h"
#include "tudat/simulation/environment_setup/createRelativisticTimeConverter.h"


namespace tudat
{

namespace simulation_setup
{

template< typename StateScalarType, typename TimeType >
void setRelativisticTimeConverter(
        const std::shared_ptr< DirectRelativisticTimeConverterSettings< StateScalarType, TimeType > >& conversionSettings,
        const SystemOfBodies& bodyMap )
{
    std::vector< std::shared_ptr< propagators::SingleArcPropagatorSettings< double, double > > > propagatorSettingsList;

    const auto topocentricConversions =
        conversionSettings->getBodyCentricToTopocentricConversionSettings( );

    for ( const auto& settings : topocentricConversions )
    {
        if ( settings->getRelativisticStateDerivativeType( ) != propagators::first_order_bodycentric_to_topocentric )
        {
            throw std::runtime_error(
                "Error in setRelativisticTimeConverter: inconsistent derivative type for topocentric conversion." );
        }

        propagatorSettingsList.push_back( settings );
    }

    propagatorSettingsList.push_back( conversionSettings->getBaryCentricToBodyCentricConversionSettings( ) );

    auto terminationSettings =
        conversionSettings->getBaryCentricToBodyCentricConversionSettings( )->getTerminationSettings( );

    const double initialTime =
        conversionSettings->getBaryCentricToBodyCentricConversionSettings( )->getInitialTime( );

    if( !std::isfinite( initialTime ) )
    {
        throw std::runtime_error( "Error in setRelativisticTimeConverter: initial propagation time is not finite." );
    }

    conversionSettings->getNumericalIntegrationSettings( )->initialTimeDeprecated_ = initialTime;

    auto multiTypeSettings = std::make_shared< propagators::MultiTypePropagatorSettings< double > >(
        propagatorSettingsList,
        conversionSettings->getNumericalIntegrationSettings( ),
        initialTime,
        terminationSettings );
    std::cerr<<"Initial combined state:\n"<<multiTypeSettings->getInitialStates()<<std::endl;

    propagators::SingleArcDynamicsSimulator< double, double > simulator(
        bodyMap,
        conversionSettings->getNumericalIntegrationSettings( ),
        multiTypeSettings,
        true,
        false,
        true );
}

template< typename StateScalarType, typename TimeType >
void setRelativisticTimeConverters(
        const SystemOfBodies& bodyMap,
        const std::map< std::string, std::shared_ptr< DirectRelativisticTimeConverterSettings< StateScalarType, TimeType > > >& converterSettings )
{
    for ( const auto& [name, settings] : converterSettings )
    {
        setRelativisticTimeConverter( settings, bodyMap );
    }
}

} // namespace simulation_setup

} // namespace tudat

namespace tudat
{
namespace simulation_setup
{

template void setRelativisticTimeConverter<double, double>(
        const std::shared_ptr< DirectRelativisticTimeConverterSettings< double, double > >& conversionSettings,
        const SystemOfBodies& bodyMap );

template void setRelativisticTimeConverters<double, double>(
        const SystemOfBodies& bodyMap,
        const std::map< std::string, std::shared_ptr< DirectRelativisticTimeConverterSettings< double, double > > >& converterSettings );

} // namespace simulation_setup
} // namespace tudat
