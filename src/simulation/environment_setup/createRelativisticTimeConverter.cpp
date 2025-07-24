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
    std::vector< std::shared_ptr< propagators::PropagatorSettings< double > > > propagatorSettingsList;

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

    std::map< propagators::IntegratedStateType, std::vector< std::shared_ptr< propagators::PropagatorSettings< double > > > > propagatorSettingsMap;
    propagatorSettingsMap[ propagators::proper_time ] = propagatorSettingsList;

    propagators::SingleArcDynamicsSimulator< double, double > simulator(
        bodyMap,
        conversionSettings->getNumericalIntegrationSettings( ),
        std::make_shared< propagators::MultiTypePropagatorSettings< double > >( propagatorSettingsMap )
    );
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
