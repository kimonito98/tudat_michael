/*    Copyright (c) 2010-2019, Delft University of Technology
 *    All rigths reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 *
 */


#include "tudat/astro/basic_astro/celestialBodyConstants.h"
#include "tudat/math/basic/coordinateConversions.h"
#include "tudat/astro/low_thrust/shape_based/sphericalShapingLeg.h"

namespace tudat
{
namespace shape_based_methods
{


//! Constructor for spherical shaping.
    SphericalShapingLeg::SphericalShapingLeg(const std::shared_ptr<ephemerides::Ephemeris> departureBodyEphemeris,
                                             const std::shared_ptr<ephemerides::Ephemeris> arrivalBodyEphemeris,
                                             const double centralBodyGravitationalParameter,
                                             const int numberOfRevolutions,
                                             const std::shared_ptr<root_finders::RootFinderSettings> rootFinderSettings,
                                             const double initialValueFreeCoefficient,
                                             const double lowerBoundFreeCoefficient,
                                             const double upperBoundFreeCoefficient) :
    mission_segments::TransferLeg( departureBodyEphemeris, arrivalBodyEphemeris, mission_segments::spherical_shaping_low_thrust_leg ),
    centralBodyGravitationalParameter_( centralBodyGravitationalParameter ),
    numberOfRevolutions_( numberOfRevolutions ),
    rootFinderSettings_( rootFinderSettings ),
    initialValueFreeCoefficient_( initialValueFreeCoefficient ),
    lowerBoundFreeCoefficient_( lowerBoundFreeCoefficient ),
    upperBoundFreeCoefficient_( upperBoundFreeCoefficient )
{
    // Normalize the gravitational parameter of the central body.
    centralBodyGravitationalParameter_ = centralBodyGravitationalParameter * std::pow( physical_constants::JULIAN_YEAR, 2.0 )
            / std::pow( physical_constants::ASTRONOMICAL_UNIT, 3.0 );

    // Define coefficients for radial distance and elevation angle composite functions.
    coefficientsRadialDistanceFunction_.setConstant( 1.0 );
    coefficientsElevationAngleFunction_.setConstant( 1.0 );

    radialDistanceCompositeFunction_ = std::make_shared< CompositeRadialFunctionSphericalShaping >(
                Eigen::VectorXd( coefficientsRadialDistanceFunction_ ) );
    elevationAngleCompositeFunction_ = std::make_shared< CompositeElevationFunctionSphericalShaping >(
                Eigen::VectorXd( coefficientsElevationAngleFunction_ ) );

}

void SphericalShapingLeg::computeTransfer( )
{
    updateDepartureAndArrivalBodies( legParameters_( 0 ), legParameters_( 1 ) );

    if( legParameters_.rows( ) != 2 )
    {
        throw std::runtime_error( "Error when updating spherical shaping object, number of inputs is inconsistent" );
    }

    // Normalize the initial state.
    Eigen::Vector6d normalizedDepartureBodyState_;
    normalizedDepartureBodyState_.segment( 0, 3 ) =
            departureBodyState_.segment( 0, 3 ) / physical_constants::ASTRONOMICAL_UNIT;
    normalizedDepartureBodyState_.segment( 3, 3 ) =
            departureBodyState_.segment( 3, 3 ) * physical_constants::JULIAN_YEAR / physical_constants::ASTRONOMICAL_UNIT;

    // Normalize the final state.
    Eigen::Vector6d normalizedArrivalBodyState_;
    normalizedArrivalBodyState_.segment( 0, 3 ) =
            arrivalBodyState_.segment( 0, 3 ) / physical_constants::ASTRONOMICAL_UNIT;
    normalizedArrivalBodyState_.segment( 3, 3 ) =
            arrivalBodyState_.segment( 3, 3 ) * physical_constants::JULIAN_YEAR / physical_constants::ASTRONOMICAL_UNIT;

    // Compute initial and final state in spherical coordinates.
    initialStateSphericalCoordinates_ = coordinate_conversions::convertCartesianToSphericalState( normalizedDepartureBodyState_ );
    if ( initialStateSphericalCoordinates_( 1 ) < 0.0 )
        initialStateSphericalCoordinates_( 1 ) += 2.0 * mathematical_constants::PI;
    finalStateSphericalCoordinates_ = coordinate_conversions::convertCartesianToSphericalState( normalizedArrivalBodyState_ );
    if ( finalStateSphericalCoordinates_( 1 ) < 0.0 )
        finalStateSphericalCoordinates_( 1 ) += 2.0 * mathematical_constants::PI;

    // Retrieve the initial value of the azimuth angle.
    initialAzimuthAngle_ = initialStateSphericalCoordinates_[ 1 ];

    // Compute final value of the azimuth angle.
    ( ( finalStateSphericalCoordinates_( 1 ) - initialStateSphericalCoordinates_( 1 ) ) < 0.0 ) ?
          ( finalAzimuthAngle_ = finalStateSphericalCoordinates_( 1 ) + 2.0 * mathematical_constants::PI * ( numberOfRevolutions_ + 1.0 ) ) :
          ( finalAzimuthAngle_ = finalStateSphericalCoordinates_( 1 ) + 2.0 * mathematical_constants::PI * ( numberOfRevolutions_ ) );


    // Compute initial and final values of the derivative of the azimuth angle w.r.t. time.
    double initialDerivativeAzimuthAngle = initialStateSphericalCoordinates_[ 4 ]
            / ( initialStateSphericalCoordinates_[ 0 ] * std::cos( initialStateSphericalCoordinates_[ 2 ] ) );
    double finalDerivativeAzimuthAngle = finalStateSphericalCoordinates_[ 4 ]
            / ( finalStateSphericalCoordinates_[ 0 ] * std::cos( finalStateSphericalCoordinates_[ 2 ] ) );

    // Compute initial and final state parametrized by azimuth angle theta.
    initialStateParametrizedByAzimuthAngle_ = ( Eigen::Vector6d() << initialStateSphericalCoordinates_[ 0 ],
            initialStateSphericalCoordinates_[ 1 ],
            initialStateSphericalCoordinates_[ 2 ],
            initialStateSphericalCoordinates_[ 3 ] / initialDerivativeAzimuthAngle,
            initialStateSphericalCoordinates_[ 4 ] / initialDerivativeAzimuthAngle,
            initialStateSphericalCoordinates_[ 5 ] / initialDerivativeAzimuthAngle ).finished();
    finalStateParametrizedByAzimuthAngle_ = ( Eigen::Vector6d() << finalStateSphericalCoordinates_[ 0 ],
            finalStateSphericalCoordinates_[ 1 ],
            finalStateSphericalCoordinates_[ 2 ],
            finalStateSphericalCoordinates_[ 3 ] / finalDerivativeAzimuthAngle,
            finalStateSphericalCoordinates_[ 4 ] / finalDerivativeAzimuthAngle,
            finalStateSphericalCoordinates_[ 5 ] / finalDerivativeAzimuthAngle ).finished();

    // Define settings for numerical quadrature, to be used to compute time of flight and final deltaV.
    quadratureSettings_ = std::make_shared< numerical_quadrature::GaussianQuadratureSettings < double > >( initialAzimuthAngle_, 16 );

    // Update value of boundary conditions of free coefficient a2
    // computeFreeCoefficientBoundaries();

    // Iterate on the free coefficient value until the time of flight matches its required value.
    // For each free coefficient value: get coefficients of radial distance and elevation angle functions to meet
    // boundary conditions, compute time of flight, compute error with respect to required time of flight
    iterateToMatchRequiredTimeOfFlight( rootFinderSettings_, lowerBoundFreeCoefficient_, upperBoundFreeCoefficient_, initialValueFreeCoefficient_ );

    // Retrieve initial step size.
    double initialStepSize = 86400.0;//integratorSettings->initialTimeStep_;

    std::cerr<<"Warning, initialStepSize set to arbitrary value"<<std::endl;

    // Vector of azimuth angles at which the time should be computed.
    Eigen::VectorXd azimuthAnglesToComputeAssociatedEpochs =
            Eigen::VectorXd::LinSpaced( std::ceil( computeNormalizedTimeOfFlight() * physical_constants::JULIAN_YEAR / initialStepSize ),
                                        initialAzimuthAngle_, finalAzimuthAngle_ );

    std::map< double, double > dataToInterpolate;
    for ( int i = 0 ; i < azimuthAnglesToComputeAssociatedEpochs.size() ; i++ )
    {
        dataToInterpolate[ convertAzimuthToTime( azimuthAnglesToComputeAssociatedEpochs[ i ] ) * physical_constants::JULIAN_YEAR ]
                = azimuthAnglesToComputeAssociatedEpochs[ i ];
    }

    // Create interpolator.
    std::shared_ptr< interpolators::InterpolatorSettings > interpolatorSettings =
            std::make_shared< interpolators::LagrangeInterpolatorSettings >( 10 );

    interpolator_ = interpolators::createOneDimensionalInterpolator( dataToInterpolate, interpolatorSettings );

    legTotalDeltaV_ = computeDeltaV( );
}

//! Convert time to azimuth (azimuth is the independent variable used in the spherical shaping method).
double SphericalShapingLeg::convertTimeToAzimuth(const double timeSinceDeparture )
{
    return  interpolator_->interpolate(timeSinceDeparture );
}

//! Convert azimuth to time (azimuth is the independent variable used in the spherical shaping method).
double SphericalShapingLeg::convertAzimuthToTime( const double currentAzimuthAngle )
{
    // Define the derivative of the time function w.r.t the currentAzimuthAngle angle theta.
    std::function< double( const double ) > derivativeTimeFunction = [ = ] ( const double currentAzimuthAngle ){

        double scalarFunctionTimeEquation = computeScalarFunctionD(currentAzimuthAngle);

        // Check that the trajectory is feasible, ie curved toward the central body.
        if ( scalarFunctionTimeEquation < 0.0 )
        {
            throw std::runtime_error ( "Error, trajectory not curved toward the central body, and thus not feasible." );
        }
        else
        {
            return std::sqrt(computeScalarFunctionD(currentAzimuthAngle) *
                             std::pow( radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle ), 2.0 )
                             / centralBodyGravitationalParameter_ );
        }
    };

    // Create numerical quadrature from quadrature settings.
    std::shared_ptr< numerical_quadrature::NumericalQuadrature< double, double > > quadrature =
            numerical_quadrature::createQuadrature(derivativeTimeFunction, quadratureSettings_, currentAzimuthAngle );

    return quadrature->getQuadrature( );
}

Eigen::MatrixXd SphericalShapingLeg::computeInverseMatrixBoundaryConditions( )
{
    Eigen::MatrixXd matrixBoundaryConditions = Eigen::MatrixXd::Zero( 10, 10 );

    for ( int i = 0 ; i < 6 ; i++ )
    {
        int index = i;
        if ( i >= 2 )
        {
            index = i + 1;
        }
        matrixBoundaryConditions( 0, i ) = radialDistanceCompositeFunction_->getComponentFunctionCurrentValue( index, initialAzimuthAngle_ );
        matrixBoundaryConditions( 1, i ) = radialDistanceCompositeFunction_->getComponentFunctionCurrentValue( index, finalAzimuthAngle_ );
        matrixBoundaryConditions( 2, i ) = radialDistanceCompositeFunction_->getComponentFunctionFirstDerivative( index, initialAzimuthAngle_ );
        matrixBoundaryConditions( 3, i ) = radialDistanceCompositeFunction_->getComponentFunctionFirstDerivative( index, finalAzimuthAngle_ );
        matrixBoundaryConditions( 4, i ) = - std::pow( initialStateSphericalCoordinates_[ 0 ], 2.0 )
                * radialDistanceCompositeFunction_->getComponentFunctionSecondDerivative( index, initialAzimuthAngle_ );
        matrixBoundaryConditions( 5, i ) = - std::pow( finalStateSphericalCoordinates_[ 0 ], 2.0 )
                * radialDistanceCompositeFunction_->getComponentFunctionSecondDerivative( index, finalAzimuthAngle_ );
    }

    // Compute value of variable alpha at initial time.
    double initialValueAlpha = computeValueConstantAlpha(initialStateParametrizedByAzimuthAngle_);

    // Compute value of variable alpha at final time.
    double finalValueAlpha = computeValueConstantAlpha(finalStateParametrizedByAzimuthAngle_);

    for ( int i = 0 ; i < 4 ; i++ )
    {
        matrixBoundaryConditions( 4, i + 6 ) =
                initialValueAlpha * elevationAngleCompositeFunction_->getComponentFunctionSecondDerivative( i, initialAzimuthAngle_ );
        matrixBoundaryConditions( 5, i + 6 ) =
                finalValueAlpha * elevationAngleCompositeFunction_->getComponentFunctionSecondDerivative( i, finalAzimuthAngle_ );
        matrixBoundaryConditions( 6, i + 6 ) = elevationAngleCompositeFunction_->getComponentFunctionCurrentValue( i, initialAzimuthAngle_ );
        matrixBoundaryConditions( 7, i + 6 ) = elevationAngleCompositeFunction_->getComponentFunctionCurrentValue( i, finalAzimuthAngle_ );
        matrixBoundaryConditions( 8, i + 6 ) = elevationAngleCompositeFunction_->getComponentFunctionFirstDerivative( i, initialAzimuthAngle_ );
        matrixBoundaryConditions( 9, i + 6 ) = elevationAngleCompositeFunction_->getComponentFunctionFirstDerivative( i, finalAzimuthAngle_ );
    }

    // Compute and return the inverse of the boundary conditions matrix.
    return matrixBoundaryConditions.inverse();
}

double SphericalShapingLeg::computeValueConstantAlpha(Eigen::Vector6d stateParametrizedByAzimuthAngle )
{
    return - ( stateParametrizedByAzimuthAngle[ 3 ] * stateParametrizedByAzimuthAngle[ 5 ] / stateParametrizedByAzimuthAngle[ 0 ] )
           / ( std::pow( stateParametrizedByAzimuthAngle[ 5 ] / stateParametrizedByAzimuthAngle[ 0 ], 2.0 )
               + std::pow( std::cos( stateParametrizedByAzimuthAngle[ 2 ] ), 2.0 ) );
}

double SphericalShapingLeg::computeValueConstantC (Eigen::Vector6d stateParametrizedByAzimuthAngle,
                                                   Eigen::Vector6d stateSphericalCoordinates)
{
    double radialDistance = stateParametrizedByAzimuthAngle[ 0 ];
    double elevationAngle = stateParametrizedByAzimuthAngle[ 2 ];
    double derivativeRadialDistance = stateParametrizedByAzimuthAngle[ 3 ];
    double derivativeElevationAngle = stateParametrizedByAzimuthAngle[ 5 ] / stateParametrizedByAzimuthAngle[ 0 ];
    double derivativeOfTimeWrtAzimuthAngle = ( stateParametrizedByAzimuthAngle[ 0 ] * std::cos( stateParametrizedByAzimuthAngle[ 2 ] ) )
                                             / stateSphericalCoordinates[ 4 ];

    return - centralBodyGravitationalParameter_ * std::pow( derivativeOfTimeWrtAzimuthAngle, 2.0 ) / std::pow( radialDistance, 2.0 )
           + 2.0 * std::pow( derivativeRadialDistance, 2.0 ) / radialDistance
           + radialDistance * ( std::pow( derivativeElevationAngle, 2.0 ) + std::pow( std::cos( elevationAngle ), 2.0 ) )
           - derivativeRadialDistance * derivativeElevationAngle * ( std::sin( elevationAngle ) * std::cos( elevationAngle ) )
             / ( std::pow( derivativeElevationAngle, 2.0 ) + std::pow( std::cos( elevationAngle ), 2.0 ) );
}

void SphericalShapingLeg::satisfyBoundaryConditions( double freeCoefficient )
{
    Eigen::VectorXd vectorBoundaryValues;
    vectorBoundaryValues.resize( 10 );

    vectorBoundaryValues[ 0 ] = 1.0 / initialStateParametrizedByAzimuthAngle_[ 0 ];
    vectorBoundaryValues[ 1 ] = 1.0 / finalStateParametrizedByAzimuthAngle_[ 0 ];
    vectorBoundaryValues[ 2 ] = - initialStateParametrizedByAzimuthAngle_[ 3 ] / std::pow( initialStateParametrizedByAzimuthAngle_[ 0 ], 2.0 );
    vectorBoundaryValues[ 3 ] = - finalStateParametrizedByAzimuthAngle_[ 3 ] / std::pow( finalStateParametrizedByAzimuthAngle_[ 0 ], 2.0 );
    vectorBoundaryValues[ 4 ] =
            computeValueConstantC(initialStateParametrizedByAzimuthAngle_, initialStateSphericalCoordinates_)
            - 2.0 * std::pow( initialStateParametrizedByAzimuthAngle_[ 3 ], 2.0 ) / initialStateParametrizedByAzimuthAngle_[ 0 ];
    vectorBoundaryValues[ 5 ] =
            computeValueConstantC(finalStateParametrizedByAzimuthAngle_, finalStateSphericalCoordinates_)
            - 2.0 * std::pow( finalStateParametrizedByAzimuthAngle_[ 3 ], 2.0 ) / finalStateParametrizedByAzimuthAngle_[ 0 ];
    vectorBoundaryValues[ 6 ] = initialStateParametrizedByAzimuthAngle_[ 2 ];
    vectorBoundaryValues[ 7 ] = finalStateParametrizedByAzimuthAngle_[ 2 ];
    vectorBoundaryValues[ 8 ] = initialStateParametrizedByAzimuthAngle_[ 5 ] / initialStateParametrizedByAzimuthAngle_[ 0 ];
    vectorBoundaryValues[ 9 ] = finalStateParametrizedByAzimuthAngle_[ 5 ] / finalStateParametrizedByAzimuthAngle_[ 0 ];

    Eigen::VectorXd vectorSecondComponentContribution;
    vectorSecondComponentContribution.resize( 10.0 );

    vectorSecondComponentContribution[ 0 ] = radialDistanceCompositeFunction_->getComponentFunctionCurrentValue( 2, initialAzimuthAngle_ );
    vectorSecondComponentContribution[ 1 ] = radialDistanceCompositeFunction_->getComponentFunctionCurrentValue( 2, finalAzimuthAngle_ );
    vectorSecondComponentContribution[ 2 ] = radialDistanceCompositeFunction_->getComponentFunctionFirstDerivative( 2, initialAzimuthAngle_ );
    vectorSecondComponentContribution[ 3 ] = radialDistanceCompositeFunction_->getComponentFunctionFirstDerivative( 2, finalAzimuthAngle_ );
    vectorSecondComponentContribution[ 4 ] = - std::pow( initialStateParametrizedByAzimuthAngle_[ 0 ], 2 )
            * radialDistanceCompositeFunction_->getComponentFunctionSecondDerivative( 2, initialAzimuthAngle_ );
    vectorSecondComponentContribution[ 5 ] = - std::pow( finalStateParametrizedByAzimuthAngle_[ 0 ], 2 )
            * radialDistanceCompositeFunction_->getComponentFunctionSecondDerivative( 2, finalAzimuthAngle_ );
    vectorSecondComponentContribution[ 6 ] = 0.0;
    vectorSecondComponentContribution[ 7 ] = 0.0;
    vectorSecondComponentContribution[ 8 ] = 0.0;
    vectorSecondComponentContribution[ 9 ] = 0.0;

    vectorSecondComponentContribution *= freeCoefficient;

    Eigen::MatrixXd inverseMatrixBoundaryConditions = computeInverseMatrixBoundaryConditions();

    Eigen::MatrixXd compositeFunctionCoefficients = inverseMatrixBoundaryConditions * ( vectorBoundaryValues - vectorSecondComponentContribution );

    for ( int i = 0 ; i < 6 ; i++ )
    {
        if ( i < 2 )
        {
            coefficientsRadialDistanceFunction_( i ) = compositeFunctionCoefficients( i );
        }
        else
        {
            coefficientsRadialDistanceFunction_( i + 1 ) = compositeFunctionCoefficients( i );
        }
    }
    coefficientsRadialDistanceFunction_( 2 ) = freeCoefficient;

    for ( int i = 0 ; i < 4 ; i++ )
    {
        coefficientsElevationAngleFunction_( i ) = compositeFunctionCoefficients( i + 6 );
    }

    radialDistanceCompositeFunction_->resetCompositeFunctionCoefficients( coefficientsRadialDistanceFunction_ );
    elevationAngleCompositeFunction_->resetCompositeFunctionCoefficients( coefficientsElevationAngleFunction_ );

}

//void SphericalShapingLeg::computeFreeCoefficientBoundaries()
//{
//    Eigen::MatrixXd Ainverse = computeInverseMatrixBoundaryConditions( );
//
//    // B-matrix
//    Eigen::VectorXd vectorBoundaryValues;
//    vectorBoundaryValues.resize( 10 );
//
//    vectorBoundaryValues[ 0 ] = 1.0 / initialStateParametrizedByAzimuthAngle_[ 0 ];
//    vectorBoundaryValues[ 1 ] = 1.0 / finalStateParametrizedByAzimuthAngle_[ 0 ];
//    vectorBoundaryValues[ 2 ] = - initialStateParametrizedByAzimuthAngle_[ 3 ] / std::pow( initialStateParametrizedByAzimuthAngle_[ 0 ], 2.0 );
//    vectorBoundaryValues[ 3 ] = - finalStateParametrizedByAzimuthAngle_[ 3 ] / std::pow( finalStateParametrizedByAzimuthAngle_[ 0 ], 2.0 );
//    vectorBoundaryValues[ 4 ] = computeInitialValueBoundariesConstant()
//                                - 2.0 * std::pow( initialStateParametrizedByAzimuthAngle_[ 3 ], 2.0 ) / initialStateParametrizedByAzimuthAngle_[ 0 ];
//    vectorBoundaryValues[ 5 ] = computeFinalValueBoundariesConstant()
//                                - 2.0 * std::pow( finalStateParametrizedByAzimuthAngle_[ 3 ], 2.0 ) / finalStateParametrizedByAzimuthAngle_[ 0 ];
//    vectorBoundaryValues[ 6 ] = initialStateParametrizedByAzimuthAngle_[ 2 ];
//    vectorBoundaryValues[ 7 ] = finalStateParametrizedByAzimuthAngle_[ 2 ];
//    vectorBoundaryValues[ 8 ] = initialStateParametrizedByAzimuthAngle_[ 5 ] / initialStateParametrizedByAzimuthAngle_[ 0 ];
//    vectorBoundaryValues[ 9 ] = finalStateParametrizedByAzimuthAngle_[ 5 ] / finalStateParametrizedByAzimuthAngle_[ 0 ];
//
//    // Aa2-matrix (no a2)
//    Eigen::VectorXd vectorSecondComponentContribution;
//    vectorSecondComponentContribution.resize( 10.0 );
//
//    vectorSecondComponentContribution[ 0 ] = radialDistanceCompositeFunction_->getComponentFunctionCurrentValue( 2, initialAzimuthAngle_ );
//    vectorSecondComponentContribution[ 1 ] = radialDistanceCompositeFunction_->getComponentFunctionCurrentValue( 2, finalAzimuthAngle_ );
//    vectorSecondComponentContribution[ 2 ] = radialDistanceCompositeFunction_->getComponentFunctionFirstDerivative( 2, initialAzimuthAngle_ );
//    vectorSecondComponentContribution[ 3 ] = radialDistanceCompositeFunction_->getComponentFunctionFirstDerivative( 2, finalAzimuthAngle_ );
//    vectorSecondComponentContribution[ 4 ] = - std::pow( initialStateParametrizedByAzimuthAngle_[ 0 ], 2 )
//                                             * radialDistanceCompositeFunction_->getComponentFunctionSecondDerivative( 2, initialAzimuthAngle_ );
//    vectorSecondComponentContribution[ 5 ] = - std::pow( finalStateParametrizedByAzimuthAngle_[ 0 ], 2 )
//                                             * radialDistanceCompositeFunction_->getComponentFunctionSecondDerivative( 2, finalAzimuthAngle_ );
//    vectorSecondComponentContribution[ 6 ] = 0.0;
//    vectorSecondComponentContribution[ 7 ] = 0.0;
//    vectorSecondComponentContribution[ 8 ] = 0.0;
//    vectorSecondComponentContribution[ 9 ] = 0.0;
//
//    // (A^-1)*B
//    Eigen::VectorXd V1 = Ainverse*vectorBoundaryValues;
//    // (A^-1)*Aa2
//    Eigen::VectorXd V2 = Ainverse*vectorSecondComponentContribution;
//
//    // 6x1
//    Eigen::Vector6d V1sliceTop = V1.head(6);
//    Eigen::Vector6d V2sliceTop = V2.head(6);
//    // 4x1
//    Eigen::Vector6d V1sliceBot = V1.tail(4);
//    Eigen::Vector6d V2sliceBot = V2.tail(4);
//
//    // loading computed coefficients
//    coefficientsElevationAngleFunction_[ 0 ] = V1sliceBot[0] - V2sliceBot[0];
//    coefficientsElevationAngleFunction_[ 1 ] = V1sliceBot[1] - V2sliceBot[1];
//    coefficientsElevationAngleFunction_[ 2 ] = V1sliceBot[2] - V2sliceBot[2];
//    coefficientsElevationAngleFunction_[ 3 ] = V1sliceBot[3] - V2sliceBot[3];
//
//
//    elevationAngleCompositeFunction_->resetCompositeFunctionCoefficients( coefficientsElevationAngleFunction_ );
//
//
//    double currentAzimuthAngle;
//    double stepsize = (finalAzimuthAngle_ - initialAzimuthAngle_)/500;
//
//    double freeCoefficientLowerBound = -100.0;
//    double freeCoefficientUpperBound =  100.0;
//
//
//    for ( currentAzimuthAngle = initialAzimuthAngle_ ; currentAzimuthAngle < finalAzimuthAngle_ ; currentAzimuthAngle = currentAzimuthAngle + stepsize ) // ignore last step gives a very high number and wrong
//    {
//        // C1, C2, C3
//        double elevationFunctionValue            = elevationAngleCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );
//        double firstDerivativeElevationFunction  = elevationAngleCompositeFunction_->evaluateCompositeFunctionFirstDerivative( currentAzimuthAngle );
//        double secondDerivativeElevationFunction = elevationAngleCompositeFunction_->evaluateCompositeFunctionSecondDerivative( currentAzimuthAngle );
//
//        double C1 = firstDerivativeElevationFunction*((secondDerivativeElevationFunction - std::sin(elevationFunctionValue)*std::cos(elevationFunctionValue))
//                                                      /(std::pow(firstDerivativeElevationFunction,2.0) + std::pow(std::cos(elevationFunctionValue),2.0)));
//
//        double C2 = std::pow(firstDerivativeElevationFunction,2.0) + std::pow(std::cos(elevationFunctionValue),2.0);
//
//
//        //Rk (no a2 comp)
//        Eigen::Vector6d RkLess;
//        RkLess[0] = 1.0;
//        RkLess[1] = currentAzimuthAngle;
//        RkLess[2] = std::cos(currentAzimuthAngle);
//        RkLess[3] = currentAzimuthAngle * std::cos(currentAzimuthAngle);
//        RkLess[4] = std::sin(currentAzimuthAngle);
//        RkLess[5] = currentAzimuthAngle * std::sin(currentAzimuthAngle);
//
//        //Rk' (no a2 comp)
//        Eigen::Vector6d RkFirstLess;
//        RkFirstLess[0] = 0.0;
//        RkFirstLess[1] = 1.0;
//        RkFirstLess[2] = -std::sin(currentAzimuthAngle);
//        RkFirstLess[3] =  std::cos(currentAzimuthAngle) - currentAzimuthAngle*std::sin(currentAzimuthAngle);
//        RkFirstLess[4] =  std::cos(currentAzimuthAngle);
//        RkFirstLess[5] =  std::sin(currentAzimuthAngle) + currentAzimuthAngle*std::cos(currentAzimuthAngle);
//
//        //Rk'' (no a2 comp)
//        Eigen::Vector6d RkSecondLess;
//        RkSecondLess[0] = 0.0;
//        RkSecondLess[1] = 0.0;
//        RkSecondLess[2] = -std::cos(currentAzimuthAngle);
//        RkSecondLess[3] = -2.0*std::sin(currentAzimuthAngle) - currentAzimuthAngle*std::cos(currentAzimuthAngle);
//        RkSecondLess[4] = -std::sin(currentAzimuthAngle);
//        RkSecondLess[5] =  2.0*std::cos(currentAzimuthAngle) - currentAzimuthAngle*std::sin(currentAzimuthAngle);
//
//        // (Rk''*(A^-1)*B)
//        double CA = RkSecondLess.transpose()*V1sliceTop;
//        // (Rk''*(A^-1)*Aa2)
//        double CB = RkSecondLess.transpose()*V2sliceTop;
//
//        // (Rk'*(A^-1)*B)
//        double CC = C1*RkFirstLess.transpose()*V1sliceTop;
//        // (Rk'*(A^-1)*Aa2)
//        double CD = C1*RkFirstLess.transpose()*V2sliceTop;
//
//        // 2*theta*C1
//        double CE = 2.0*currentAzimuthAngle*C1;
//
//        // (Rk*(A^-1)*B)
//        double CF = RkLess.transpose()*V1sliceTop;
//        // (Rk*(A^-1)*Aa2)
//        double CG = RkLess.transpose()*V2sliceTop;
//
//        double numerator         = C2*CF + CA - CC;
//        double secondDenominator = C2*CG - C2*std::pow(currentAzimuthAngle,2.0) + CB - 2 - CD + CE;
//
//        double freeCoefficient = numerator/secondDenominator;
//
//
//        double firstDenominator = CF - CG*freeCoefficient + freeCoefficient*std::pow(currentAzimuthAngle,2.0);
//
//        coefficientsRadialDistanceFunction_[0] = V1[0] - freeCoefficient*V2[0];
//        coefficientsRadialDistanceFunction_[1] = V1[1] - freeCoefficient*V2[1];
//        coefficientsRadialDistanceFunction_[2] = freeCoefficient;
//        coefficientsRadialDistanceFunction_[3] = V1[2] - freeCoefficient*V2[2];
//        coefficientsRadialDistanceFunction_[4] = V1[3] - freeCoefficient*V2[3];
//        coefficientsRadialDistanceFunction_[5] = V1[4] - freeCoefficient*V2[4];
//        coefficientsRadialDistanceFunction_[6] = V1[5] - freeCoefficient*V2[5];
//
//
//        radialDistanceCompositeFunction_->resetCompositeFunctionCoefficients( coefficientsRadialDistanceFunction_ );
//
//        double radiusFunctionValue            = radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );
//
//
//
//        int zeroFlag = 1;
//        // fix numbers close to zero that influence equality sign change
//        if (std::abs(firstDenominator) < 1e-10)
//        {
//            firstDenominator = 0.0;
//            zeroFlag = 0;
//
//        }
//        if (std::abs(secondDenominator) < 1e-10)
//        {
//            secondDenominator = 0.0;
//            zeroFlag = 0;
//        }
//
//        if ( ((firstDenominator < 0 && secondDenominator > 0 && radiusFunctionValue > 0) ||
//              (firstDenominator > 0 && secondDenominator < 0 && radiusFunctionValue > 0) ||
//              (firstDenominator > 0 && secondDenominator > 0 && radiusFunctionValue < 0) ||
//              (firstDenominator < 0 && secondDenominator < 0 && radiusFunctionValue < 0))  && (zeroFlag != 0.0) ) //
//        {
//            if (freeCoefficient > freeCoefficientLowerBound)
//            {
//                freeCoefficientLowerBound = freeCoefficient + 0.001*std::abs(freeCoefficient); // to ensure the boundary value itself does not lead to TOF = nan
//            }
//        }
//        else
//        {
//            if (zeroFlag != 0.0)
//            {
//                if (freeCoefficient < freeCoefficientUpperBound)
//                {
//                    freeCoefficientUpperBound = freeCoefficient - 0.001*std::abs(freeCoefficient);
//                }
//            }
//        }
//    }
//
//    lowerBoundFreeCoefficient_ = freeCoefficientLowerBound;
//    upperBoundFreeCoefficient_ = freeCoefficientUpperBound;
//    initialValueFreeCoefficient_ = 0.0;
//    coefficientsRadialDistanceFunction_[ 2 ] = 0.0;
//    radialDistanceCompositeFunction_->resetCompositeFunctionCoefficients( coefficientsRadialDistanceFunction_ );
//
//}

double SphericalShapingLeg::computeScalarFunctionD(double currentAzimuthAngle )
{
    double radialFunctionValue = radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );
    double firstDerivativeRadialFunction = radialDistanceCompositeFunction_->evaluateCompositeFunctionFirstDerivative( currentAzimuthAngle );
    double secondDerivativeRadialFunction = radialDistanceCompositeFunction_->evaluateCompositeFunctionSecondDerivative( currentAzimuthAngle );

    double elevationFunctionValue = elevationAngleCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );
    double firstDerivativeElevationFunction = elevationAngleCompositeFunction_->evaluateCompositeFunctionFirstDerivative( currentAzimuthAngle );
    double secondDerivativeElevationFunction = elevationAngleCompositeFunction_->evaluateCompositeFunctionSecondDerivative( currentAzimuthAngle );

    return - secondDerivativeRadialFunction + 2.0 * std::pow( firstDerivativeRadialFunction, 2.0 ) / radialFunctionValue
            + firstDerivativeRadialFunction * firstDerivativeElevationFunction
            * ( secondDerivativeElevationFunction - std::sin( elevationFunctionValue ) * std::cos( elevationFunctionValue ) )
            / ( std::pow( firstDerivativeElevationFunction, 2.0 ) + std::pow( std::cos( elevationFunctionValue ), 2.0 ) )
            + radialFunctionValue * ( std::pow( firstDerivativeElevationFunction, 2.0 ) + std::pow( std::cos( elevationFunctionValue ), 2.0 ) );
}

double SphericalShapingLeg::computeDerivativeScalarFunctionD(double currentAzimuthAngle )
{
    double radialFunctionValue = radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );
    double firstDerivativeRadialFunction = radialDistanceCompositeFunction_->evaluateCompositeFunctionFirstDerivative( currentAzimuthAngle );
    double secondDerivativeRadialFunction = radialDistanceCompositeFunction_->evaluateCompositeFunctionSecondDerivative( currentAzimuthAngle );
    double thirdDerivativeRadialFunction = radialDistanceCompositeFunction_->evaluateCompositeFunctionThirdDerivative( currentAzimuthAngle );

    double elevationFunctionValue = elevationAngleCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );
    double firstDerivativeElevationFunction = elevationAngleCompositeFunction_->evaluateCompositeFunctionFirstDerivative( currentAzimuthAngle );
    double secondDerivativeElevationFunction = elevationAngleCompositeFunction_->evaluateCompositeFunctionSecondDerivative( currentAzimuthAngle );
    double thirdDerivativeElevationFunction = elevationAngleCompositeFunction_->evaluateCompositeFunctionThirdDerivative( currentAzimuthAngle );

    // Define constants F1, F2, F3 and F4 as proposed in... (ADD REFERENCE).
    double F1 = std::pow( firstDerivativeElevationFunction, 2.0 ) + std::pow( std::cos( elevationFunctionValue ), 2.0 );
    double F2 = secondDerivativeElevationFunction - std::sin( 2.0 * elevationFunctionValue ) / 2.0;
    double F3 = std::cos( 2.0 * elevationFunctionValue ) + 2.0 * std::pow( firstDerivativeElevationFunction, 2.0 ) + 1.0;
    double F4 = 2.0 * secondDerivativeElevationFunction - std::sin( 2.0 * elevationFunctionValue );

    return  F1 * firstDerivativeRadialFunction - thirdDerivativeRadialFunction
            - 2.0 * std::pow( firstDerivativeRadialFunction, 3.0 ) / std::pow( radialFunctionValue, 2.0 )
            + 4.0 * firstDerivativeRadialFunction * secondDerivativeRadialFunction / radialFunctionValue
            + F4 * firstDerivativeElevationFunction * radialFunctionValue
            + 2.0 * firstDerivativeElevationFunction * firstDerivativeRadialFunction
            * ( thirdDerivativeElevationFunction - firstDerivativeElevationFunction * std::cos( 2.0 * elevationFunctionValue ) ) / F3
            + F2 * firstDerivativeElevationFunction * secondDerivativeRadialFunction / F1
            + F2 * firstDerivativeRadialFunction * secondDerivativeElevationFunction / F1
            - 4.0 * F4 * F2 * std::pow( firstDerivativeElevationFunction, 2.0 ) * firstDerivativeRadialFunction / std::pow( F3, 2.0 );
}

//! Compute the time of flight
double SphericalShapingLeg::computeNormalizedTimeOfFlight()
{
    return convertAzimuthToTime(finalAzimuthAngle_);
}

//! Iterate to match the required time of flight
void SphericalShapingLeg::iterateToMatchRequiredTimeOfFlight( std::shared_ptr< root_finders::RootFinderSettings > rootFinderSettings,
                                                              const double lowerBound,
                                                              const double upperBound,
                                                              const double initialGuess )
{
    // Define the structure updating the time of flight from the free coefficient value, while still satisfying the boundary conditions.
    std::function< void( double ) > satisfyBoundaryConditionsFunction =
            std::bind( &SphericalShapingLeg::satisfyBoundaryConditions, this, std::placeholders::_1 );
    std::function< double ( ) >  computeTOFfunction =
            std::bind( &SphericalShapingLeg::computeNormalizedTimeOfFlight, this );
    std::function< double ( ) > getRequiredTOFfunction =
            std::bind( &SphericalShapingLeg::getNormalizedRequiredTimeOfFlight, this );

    std::shared_ptr< basic_mathematics::Function< double, double > > timeOfFlightFunction =
            std::make_shared< SphericalShapingLeg::TimeOfFlightFunction >(
                satisfyBoundaryConditionsFunction, computeTOFfunction, getRequiredTOFfunction );

    // Create root finder from root finder settings.
    std::shared_ptr< root_finders::RootFinder< double > > rootFinder = root_finders::createRootFinder(
                rootFinderSettings, lowerBound, upperBound, initialGuess );

    // Iterate to find the free coefficient value that matches the required time of flight.
    double updatedFreeCoefficient = rootFinder->execute( timeOfFlightFunction, initialGuess );
}

//! Compute current derivative of the azimuth angle.
double SphericalShapingLeg::computeFirstDerivativeAzimuthAngleWrtTime( const double currentAzimuthAngle )
{
    // Compute scalar function time equation.
    double scalarFunctionTimeEquation = computeScalarFunctionD(currentAzimuthAngle);

    // Compute current radial distance.
    double radialDistance = radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );

    // Compute and return the first derivative of the azimuth angle w.r.t. time.
    return std::sqrt( centralBodyGravitationalParameter_ / ( scalarFunctionTimeEquation * std::pow( radialDistance, 2.0 ) ) );
}

//! Compute second derivative of the azimuth angle.
double SphericalShapingLeg::computeSecondDerivativeAzimuthAngleWrtTime( const double currentAzimuthAngle )
{
    // Compute first derivative azimuth angle w.r.t. time.
    double firstDerivativeAzimuthAngle = computeFirstDerivativeAzimuthAngleWrtTime( currentAzimuthAngle );

    // Compute scalar function of the time equation, and its derivative w.r.t. azimuth angle.
    double scalarFunctionTimeEquation = computeScalarFunctionD(currentAzimuthAngle);
    double derivativeScalarFunctionTimeEquation = computeDerivativeScalarFunctionD(currentAzimuthAngle);

    // Compute radial distance, and its derivative w.r.t. azimuth angle.
    double radialDistance = radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );
    double firstDerivativeRadialDistance = radialDistanceCompositeFunction_->evaluateCompositeFunctionFirstDerivative( currentAzimuthAngle );

    return - std::pow( firstDerivativeAzimuthAngle, 2.0 )
            * ( derivativeScalarFunctionTimeEquation / ( 2.0 * scalarFunctionTimeEquation ) + firstDerivativeRadialDistance / radialDistance );
}


//! Compute normalized state vector in spherical coordinates.
Eigen::Vector6d SphericalShapingLeg::computeNormalizedStateInSphericalCoordinates(const double currentAzimuthAngle )
{
    Eigen::Vector6d currentSphericalState;
    currentSphericalState.segment( 0, 3 ) = ( Eigen::Vector3d() << radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle ),
                                              currentAzimuthAngle, elevationAngleCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle ) ).finished();

    // Compute first derivative of the azimuth angle w.r.t. time.
    double derivativeAzimuthAngle = computeFirstDerivativeAzimuthAngleWrtTime( currentAzimuthAngle );

    // Compute and return current velocity vector in spherical coordinates.
    currentSphericalState.segment( 3, 3 ) = derivativeAzimuthAngle *
            computeNormalizedVelocityParametrizedByAzimuthAngle(currentAzimuthAngle);

    return currentSphericalState;
}


//! Compute current cartesian state.
Eigen::Vector6d SphericalShapingLeg::computeCurrentStateVectorFromAzimuth(const double currentAzimuthAngle )
{
    Eigen::Vector6d normalizedStateVector =
            coordinate_conversions::convertSphericalToCartesianState(
                    computeNormalizedStateInSphericalCoordinates(currentAzimuthAngle) );

    Eigen::Vector6d dimensionalStateVector;
    dimensionalStateVector.segment( 0, 3 ) = normalizedStateVector.segment( 0, 3 )
            * physical_constants::ASTRONOMICAL_UNIT;
    dimensionalStateVector.segment( 3, 3 ) = normalizedStateVector.segment( 3, 3 )
            * physical_constants::ASTRONOMICAL_UNIT / physical_constants::JULIAN_YEAR;

    return dimensionalStateVector;
}

//! Compute current normalized velocity in spherical coordinates parametrized by azimuth angle theta.
Eigen::Vector3d SphericalShapingLeg::computeNormalizedVelocityParametrizedByAzimuthAngle(const double currentAzimuthAngle )
{

    // Retrieve current radial distance and elevation angle, as well as their derivatives w.r.t. azimuth angle.
    double radialDistance = radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );
    double derivativeRadialDistance = radialDistanceCompositeFunction_->evaluateCompositeFunctionFirstDerivative( currentAzimuthAngle );
    double elevationAngle = elevationAngleCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );
    double derivativeElevationAngle = elevationAngleCompositeFunction_->evaluateCompositeFunctionFirstDerivative( currentAzimuthAngle );

    // Compute and return velocity vector parametrized by azimuth angle.
    return ( Eigen::Vector3d() << derivativeRadialDistance,
             radialDistance * std::cos( elevationAngle ),
             radialDistance * derivativeElevationAngle ).finished();
}


//! Compute current normalized acceleration in spherical coordinates parametrized by azimuth angle theta.
Eigen::Vector3d SphericalShapingLeg::computeNormalizedThrustAccelerationParametrizedByAzimuthAngle(const double currentAzimuthAngle )
{
    // Retrieve spherical coordinates and their derivatives w.r.t. to the azimuth angle.
    double radialDistance = radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );
    double firstDerivativeRadialDistance = radialDistanceCompositeFunction_->evaluateCompositeFunctionFirstDerivative( currentAzimuthAngle );
    double secondDerivativeRadialDistance = radialDistanceCompositeFunction_->evaluateCompositeFunctionSecondDerivative( currentAzimuthAngle );
    double elevationAngle = elevationAngleCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );
    double firstDerivativeElevationAngle = elevationAngleCompositeFunction_->evaluateCompositeFunctionFirstDerivative( currentAzimuthAngle );
    double secondDerivativeElevationAngle = elevationAngleCompositeFunction_->evaluateCompositeFunctionSecondDerivative( currentAzimuthAngle );

    // Compute and return acceleration vector parametrized by the azimuth angle theta.
    Eigen::Vector3d accelerationParametrizedByAzimuthAngle;
    accelerationParametrizedByAzimuthAngle[ 0 ] = secondDerivativeRadialDistance
            - radialDistance * ( std::pow( firstDerivativeElevationAngle, 2.0 ) + std::pow( std::cos( elevationAngle ), 2.0 ) );
    accelerationParametrizedByAzimuthAngle[ 1 ] = 2.0 * firstDerivativeRadialDistance * std::cos( elevationAngle )
            - 2.0 * radialDistance * firstDerivativeElevationAngle * std::sin( elevationAngle );
    accelerationParametrizedByAzimuthAngle[ 2 ] = 2.0 * firstDerivativeRadialDistance * firstDerivativeElevationAngle
            + radialDistance * ( secondDerivativeElevationAngle + std::sin( elevationAngle ) * std::cos( elevationAngle ) );

    return accelerationParametrizedByAzimuthAngle;

}

//! Compute normalized thrust acceleration vector in spherical coordinates.
Eigen::Vector3d SphericalShapingLeg::computeNormalizedThrustAccelerationInSphericalCoordinates(const double currentAzimuthAngle )
{
    // Compute current radial distance.
    double radialDistance = radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle );

    // Compute first and second derivatives of the azimuth angle w.r.t. time.
    double firstDerivativeAzimuthAngleWrtTime = computeFirstDerivativeAzimuthAngleWrtTime( currentAzimuthAngle );
    double secondDerivativeAzimuthAngleWrtTime = computeSecondDerivativeAzimuthAngleWrtTime( currentAzimuthAngle );

    // Compute velocity vector parametrized by azimuth angle theta.
    Eigen::Vector3d velocityParametrizedByAzimuthAngle = computeNormalizedVelocityParametrizedByAzimuthAngle(
            currentAzimuthAngle);

    // Compute acceleration vector parametrized by azimuth angle theta.
    Eigen::Vector3d accelerationParametrizedByAzimuthAngle = computeNormalizedThrustAccelerationParametrizedByAzimuthAngle(
            currentAzimuthAngle);


    // Compute and return the current thrust acceleration vector in spherical coordinates.
    return std::pow( firstDerivativeAzimuthAngleWrtTime, 2.0 ) * accelerationParametrizedByAzimuthAngle
            + secondDerivativeAzimuthAngleWrtTime * velocityParametrizedByAzimuthAngle
            + centralBodyGravitationalParameter_ / std::pow( radialDistance, 3.0 ) * ( Eigen::Vector3d() << radialDistance, 0.0, 0.0 ).finished();

}

//! Compute current thrust acceleration in cartesian coordinates.
Eigen::Vector3d SphericalShapingLeg::computeCurrentThrustAccelerationFromAzimuth(const double currentAzimuthAngle )
{
    if( thrustAccelerationVectorCache_.count( currentAzimuthAngle ) == 0 )
    {
        Eigen::Vector6d sphericalStateToBeConverted;
        sphericalStateToBeConverted.segment( 0, 3 ) =
                ( Eigen::Vector3d() << radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle ),
                  currentAzimuthAngle, elevationAngleCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle ) ).finished();
        sphericalStateToBeConverted.segment( 3, 3 ) = computeNormalizedThrustAccelerationInSphericalCoordinates(
                currentAzimuthAngle);

        Eigen::Vector3d normalizedThrustAccelerationVector =
                coordinate_conversions::convertSphericalToCartesianState( sphericalStateToBeConverted ).segment( 3, 3 );

        thrustAccelerationVectorCache_[ currentAzimuthAngle ] = normalizedThrustAccelerationVector * physical_constants::ASTRONOMICAL_UNIT
                / std::pow( physical_constants::JULIAN_YEAR, 2.0 );
    }

    return thrustAccelerationVectorCache_.at( currentAzimuthAngle );

}

//! Compute magnitude cartesian acceleration.
double  SphericalShapingLeg::computeCurrentThrustAccelerationMagnitudeFromAzimuth( double currentAzimuthAngle )
{
    return computeCurrentThrustAccelerationFromAzimuth(currentAzimuthAngle).norm( );
}

//! Compute direction thrust acceleration in cartesian coordinates.
Eigen::Vector3d SphericalShapingLeg::computeCurrentThrustAccelerationDirectionFromAzimuth( double currentAzimuthAngle )
{
    return computeCurrentThrustAccelerationFromAzimuth(currentAzimuthAngle).normalized( );
}

//! Compute final deltaV.
double SphericalShapingLeg::computeDeltaV( )
{
    // Define function to integrate: time derivative of the deltaV multiplied by a factor which changes the variable of
    // integration from the time to the azimuth
    std::function< double( const double ) > derivativeFunctionDeltaV = [ = ] ( const double currentAzimuthAngle )
    {
        double thrustAcceleration = computeNormalizedThrustAccelerationInSphericalCoordinates(currentAzimuthAngle).norm();
        double derivativeOfTimeWithRespectToAzimuth = std::sqrt(computeScalarFunctionD(currentAzimuthAngle)
                * std::pow( radialDistanceCompositeFunction_->evaluateCompositeFunction( currentAzimuthAngle ), 2.0 )
                / centralBodyGravitationalParameter_ );

        return thrustAcceleration * derivativeOfTimeWithRespectToAzimuth;

    };

    // Define numerical quadrature from quadratrure settings.
    std::shared_ptr< numerical_quadrature::NumericalQuadrature< double, double > > quadrature =
            numerical_quadrature::createQuadrature( derivativeFunctionDeltaV, quadratureSettings_, finalAzimuthAngle_ );

    // Return dimensional deltaV
    return quadrature->getQuadrature( ) * physical_constants::ASTRONOMICAL_UNIT / physical_constants::JULIAN_YEAR;
}


} // namespace shape_based_methods
} // namespace tudat
