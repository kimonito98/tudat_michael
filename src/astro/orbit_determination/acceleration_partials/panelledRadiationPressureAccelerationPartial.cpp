#include <iostream>

#include "tudat/astro/orbit_determination/estimatable_parameters/radiationPressureCoefficient.h"
#include "tudat/astro/orbit_determination/acceleration_partials/panelledRadiationPressureAccelerationPartial.h"

namespace tudat
{

namespace acceleration_partials
{

//! Function for updating partial w.r.t. the bodies' positions.
void PanelledRadiationPressurePartial::update( const double currentTime )
{
    if( !( currentTime_ == currentTime ) )
    {
        Eigen::Vector3d inertialVectorFromSource = radiationPressureAcceleration_->getTargetPositionWrtSource( );
        Eigen::Vector3d bodyFixedUnitVectorToSource =
            - ( radiationPressureAcceleration_->getTargetRotationFromGlobalToLocalFrame( ) *
                inertialVectorFromSource.normalized( ) );
        double distanceToSource = inertialVectorFromSource.norm( );
        Eigen::Vector3d currentAcceleration = radiationPressureAcceleration_->getAcceleration( );
        double currentRadiationPressure = radiationPressureAcceleration_->getCurrentRadiationPressure();
        double currentMass = radiationPressureAcceleration_->getCurrentTargetMass( );

        currentPartialWrtPosition_.setZero( );

        if( currentRadiationPressure > 0.0  )
        {
            currentSourceUnitVectorPartial_ =  -1.0 / distanceToSource * (
                        Eigen::Matrix3d::Identity( ) - bodyFixedUnitVectorToSource * bodyFixedUnitVectorToSource.transpose( ) );
            currentRadiationPressurePositionPartial_ =
                    2.0 * currentRadiationPressure * bodyFixedUnitVectorToSource.transpose( ) / ( distanceToSource );

            currentCosineAnglePartial_ = Eigen::Matrix< double, 1, 3 >::Zero( );
            Eigen::Vector3d currentPanelReactionVector = Eigen::Vector3d::Zero( );
            Eigen::Vector3d currentPanelNormal = Eigen::Vector3d::Zero( );
            Eigen::Matrix3d currentPanelPartialContribution = Eigen::Matrix3d::Zero( );


            double currentPanelArea = 0.0, currentPanelEmissivity = 0.0, cosineOfPanelInclination = 0.0;

            for( int i = 0; i < panelledTargetModel_->getTotalNumberOfPanels( ); i++ )
            {
                currentPanelNormal = panelledTargetModel_->getSurfaceNormals( ).at( i );
                cosineOfPanelInclination = panelledTargetModel_->getSurfacePanelCosines( ).at( i );

                currentPanelPartialContribution.setZero( );
                if( cosineOfPanelInclination > 0.0 )
                {
                    currentCosineAnglePartial_ = currentPanelNormal.transpose( ) * currentSourceUnitVectorPartial_;

                    currentPanelArea = panelledTargetModel_->getBodyFixedPanels( ).at( i )->getPanelArea( );
                    currentPanelReactionVector = panelledTargetModel_->getPanelForces( ).at( i ) / ( currentRadiationPressure * currentPanelArea );

                    currentPanelPartialContribution += panelledTargetModel_->getFullPanels( ).at( i )->getReflectionLaw( )->
                        evaluateReactionVectorDerivativeWrtTargetPosition(
                            currentPanelNormal, -bodyFixedUnitVectorToSource, cosineOfPanelInclination, currentPanelReactionVector,
                            currentSourceUnitVectorPartial_, currentCosineAnglePartial_ );

                    currentPanelPartialContribution *= radiationPressureCoefficientFunction_() * currentRadiationPressure * currentPanelArea;


                }
                currentPartialWrtPosition_ += currentPanelPartialContribution;
            }

            Eigen::Matrix3d rotationToInertialFrame =
                Eigen::Matrix3d( radiationPressureAcceleration_->getTargetRotationFromLocalToGlobalFrame( ) );
            currentPartialWrtPosition_ = rotationToInertialFrame * currentPartialWrtPosition_ * rotationToInertialFrame.transpose( );
            currentPartialWrtPosition_ /= currentMass;

            currentPartialWrtPosition_ += currentAcceleration / currentRadiationPressure *
                ( currentRadiationPressurePositionPartial_ * rotationToInertialFrame.transpose( ) );

        }
        currentTime_ = currentTime;

    }
}


//! Calculates partial derivative of panelled radiation pressure acceleration wrt radiation pressure coefficient.
Eigen::Vector3d computePartialOfPaneledRadiationPressureAccelerationWrtRadiationPressureCoefficient(
    Eigen::Vector3d currentAcceleration, double radiationPressureCoefficient )
{
    return currentAcceleration / radiationPressureCoefficient ;
}

void PanelledRadiationPressurePartial::wrtRadiationPressureCoefficient( Eigen::MatrixXd& partial )
{
    partial = computePartialOfPaneledRadiationPressureAccelerationWrtRadiationPressureCoefficient(
        radiationPressureAcceleration_->getAcceleration( ), radiationPressureCoefficientFunction_());
}

void PanelledRadiationPressurePartial::wrtDiffuseReflectivity(
        Eigen::MatrixXd& partial,
        const std::string& panelTypeId)
{
    Eigen::Vector3d partial_temp = Eigen::Vector3d::Zero();
    partial = partial_temp;

    std::function<double()> targetMassFunction = radiationPressureAcceleration_->getTargetMassFunction();
    double spacecraftMass = targetMassFunction();

    std::function<Eigen::Quaterniond()> targetRotationFromLocalToGlobalFrameFunction = radiationPressureAcceleration_->getTargetRotationFromLocalToGlobalFrameFunction();
    Eigen::Quaterniond targetRotationFromGlobalToLocalFrame = targetRotationFromLocalToGlobalFrameFunction().inverse( );
    std::function<Eigen::Vector3d()> sourceCenterPositionInGlobalFrameFunction = radiationPressureAcceleration_->getSourcePositionFunction();
    std::function<Eigen::Vector3d()> targetCenterPositionInGlobalFrameFunction = radiationPressureAcceleration_->getTargetPositionFunction();
    Eigen::Vector3d targetCenterPositionInGlobalFrame = targetCenterPositionInGlobalFrameFunction() - sourceCenterPositionInGlobalFrameFunction();
    Eigen::Vector3d sourceToTargetDirectionLocalFrame = targetRotationFromGlobalToLocalFrame * targetCenterPositionInGlobalFrame.normalized( );
    double receivedIrradiance = radiationPressureAcceleration_->getReceivedIrradiance();
    if (receivedIrradiance >= 0)
    {
        Eigen::Vector3d forcePartialWrtDiffuseReflectivity = targetRotationFromLocalToGlobalFrameFunction() *
                panelledTargetModel_->evaluateRadiationPressureForcePartialWrtDiffuseReflectivity(receivedIrradiance, sourceToTargetDirectionLocalFrame);
        partial += forcePartialWrtDiffuseReflectivity/spacecraftMass;
    }

}

void PanelledRadiationPressurePartial::wrtSpecularReflectivity(
        Eigen::MatrixXd& partial,
        const std::string& panelTypeId)
{
    Eigen::Vector3d partial_temp = Eigen::Vector3d::Zero();
    partial = partial_temp;

    std::function<double()> targetMassFunction = radiationPressureAcceleration_->getTargetMassFunction();
    double spacecraftMass = targetMassFunction();

    std::function<Eigen::Quaterniond()> targetRotationFromLocalToGlobalFrameFunction = radiationPressureAcceleration_->getTargetRotationFromLocalToGlobalFrameFunction();
    Eigen::Quaterniond targetRotationFromGlobalToLocalFrame = targetRotationFromLocalToGlobalFrameFunction().inverse( );
    std::function<Eigen::Vector3d()> sourceCenterPositionInGlobalFrameFunction = radiationPressureAcceleration_->getSourcePositionFunction();
    std::function<Eigen::Vector3d()> targetCenterPositionInGlobalFrameFunction = radiationPressureAcceleration_->getTargetPositionFunction();
    Eigen::Vector3d targetCenterPositionInGlobalFrame = targetCenterPositionInGlobalFrameFunction() - sourceCenterPositionInGlobalFrameFunction();
    Eigen::Vector3d sourceToTargetDirectionLocalFrame = targetRotationFromGlobalToLocalFrame * targetCenterPositionInGlobalFrame.normalized( );
    double receivedIrradiance = radiationPressureAcceleration_->getReceivedIrradiance();
    if (receivedIrradiance >= 0)
    {
        Eigen::Vector3d forcePartialWrtSpecularReflectivity = targetRotationFromLocalToGlobalFrameFunction() *
                panelledTargetModel_->evaluateRadiationPressureForcePartialWrtSpecularReflectivity(receivedIrradiance, sourceToTargetDirectionLocalFrame);
        partial += forcePartialWrtSpecularReflectivity/spacecraftMass;
    }

}




//! Function for setting up and retrieving a function returning a partial w.r.t. a double parameter.
std::pair< std::function< void( Eigen::MatrixXd& ) >, int > PanelledRadiationPressurePartial::getParameterPartialFunction(
        std::shared_ptr< estimatable_parameters::EstimatableParameter< double > > parameter )
{
    std::function< void( Eigen::MatrixXd& ) > partialFunction;
    int numberOfRows = 0;

    // Check if parameter dependency exists.
    if( parameter->getParameterName( ).second.first == acceleratedBody_ )
    {
        switch( parameter->getParameterName( ).first )
        {
        // Set function returning partial w.r.t. radiation pressure coefficient.
        case estimatable_parameters::radiation_pressure_coefficient:

            partialFunction = std::bind( &PanelledRadiationPressurePartial::wrtRadiationPressureCoefficient,
                                           this, std::placeholders::_1 );
            numberOfRows = 1;
            break;

        case estimatable_parameters::specular_reflectivity:
            partialFunction = std::bind( &PanelledRadiationPressurePartial::wrtSpecularReflectivity,
                                         this, std::placeholders::_1, parameter->getParameterName( ).second.second );
            numberOfRows = 1;
            break;

        case estimatable_parameters::diffuse_reflectivity:
            partialFunction = std::bind( &PanelledRadiationPressurePartial::wrtDiffuseReflectivity,
                                         this, std::placeholders::_1, parameter->getParameterName( ).second.second );
            numberOfRows = 1;
            break;

        default:
            break;
        }
    }
    return std::make_pair( partialFunction, numberOfRows );
}

//! Function for setting up and retrieving a function returning a partial w.r.t. a vector parameter.
std::pair< std::function< void( Eigen::MatrixXd& ) >, int > PanelledRadiationPressurePartial::getParameterPartialFunction(
        std::shared_ptr< estimatable_parameters::EstimatableParameter< Eigen::VectorXd > > parameter )
{
    std::function< void( Eigen::MatrixXd& ) > partialFunction;
    int numberOfRows = 0;

    // Check if parameter dependency exists.
    if( parameter->getParameterName( ).second.first == acceleratedBody_ )
    {
        switch( parameter->getParameterName( ).first )
        {
        // Set function returning partial w.r.t. radiation pressure coefficient.
        case estimatable_parameters::arc_wise_radiation_pressure_coefficient:

            if( std::dynamic_pointer_cast< estimatable_parameters::ArcWisePanelledRadiationPressureCoefficient >( parameter ) != nullptr )
            {
                partialFunction = std::bind(
                            &PanelledRadiationPressurePartial::wrtArcWiseRadiationPressureCoefficient, this, std::placeholders::_1,
                            std::dynamic_pointer_cast< estimatable_parameters::ArcWisePanelledRadiationPressureCoefficient >( parameter ) );
            }
            else
            {
                throw std::runtime_error( "Error when making radiation pressure partial, arcwise radiation pressure parameter not consistent" );
            }
            numberOfRows = parameter->getParameterSize( );

            break;
        default:
            break;
        }
    }
    return std::make_pair( partialFunction, numberOfRows );
}


}

}

