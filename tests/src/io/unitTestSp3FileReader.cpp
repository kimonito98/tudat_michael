/*    Copyright (c) 2010-2026, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <fstream>
#include <iterator>

#include <boost/filesystem.hpp>

#include "tudat/basics/testMacros.h"
#include "tudat/io/readSp3File.h"
#include "tudat/astro/basic_astro/timeConversions.h"
#include "tudat/astro/basic_astro/physicalConstants.h"

namespace tudat
{
namespace unit_tests
{

using namespace tudat;
using namespace tudat::input_output;

BOOST_AUTO_TEST_SUITE( test_sp3_file_reader )

BOOST_AUTO_TEST_CASE( testSimpleSp3Read )
{
    const boost::filesystem::path temporaryFile =
            boost::filesystem::temp_directory_path( ) / boost::filesystem::unique_path( "tudat-sp3-%%%%%%.sp3" );

    {
        std::ofstream outputFile( temporaryFile.string( ) );
        outputFile << "#cP2024 01 02 00 00 00.00000000       2   ORB IGS14 FIT HLM\n";
        outputFile << "## 2295 172800.00000000   900.00000000 60295 0.0000000000000\n";
        outputFile << "+   1    G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n";
        outputFile << "++       0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n";
        outputFile << "%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n";
        outputFile << "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n";
        outputFile << "%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n";
        outputFile << "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n";
        outputFile << "%i    0    0    0    0    0    0    0    0    0    0    0    0\n";
        outputFile << "%i    0    0    0    0    0    0    0    0    0    0    0    0\n";
        outputFile << "/* COMMENT LINE 1\n";
        outputFile << "/* COMMENT LINE 2\n";
        outputFile << "/* COMMENT LINE 3\n";
        outputFile << "/* COMMENT LINE 4\n";
        outputFile << "/* COMMENT LINE 5\n";
        outputFile << "/* COMMENT LINE 6\n";
        outputFile << "/* COMMENT LINE 7\n";
        outputFile << "/* COMMENT LINE 8\n";
        outputFile << "/* COMMENT LINE 9\n";
        outputFile << "/* COMMENT LINE 10\n";
        outputFile << "/* COMMENT LINE 11\n";
        outputFile << "/* COMMENT LINE 12\n";
        outputFile << "*  2024 01 02 00 00 00.00000000\n";
        outputFile << "PG01  12345.000000  23456.000000  34567.000000  999999.999999 999999.999999 999999.999999\n";
        outputFile << "VG01      1.000000      2.000000      3.000000  999999.999999 999999.999999 999999.999999\n";
        outputFile << "*  2024 01 02 00 15 30.00000000\n";
        outputFile << "PG01  12346.000000  23457.000000  34568.000000  999999.999999 999999.999999 999999.999999\n";
        outputFile << "VG01      4.000000      5.000000      6.000000  999999.999999 999999.999999 999999.999999\n";
        outputFile << "EOF\n";
    }

    const std::shared_ptr< Sp3FileContents > fileContents = readSp3File( temporaryFile.string( ) );
    const std::shared_ptr< SP3cFileContents > legacyFileContents = readSp3cFile( temporaryFile.string( ) );

    BOOST_CHECK_EQUAL( fileContents->frameName, "IGS14" );
    BOOST_CHECK_EQUAL( fileContents->analysisCenter, "HLM" );
    BOOST_CHECK_EQUAL( fileContents->timeScale, "GPS" );
    BOOST_CHECK_EQUAL( legacyFileContents->timeScale, "GPS" );
    BOOST_CHECK_EQUAL( fileContents->declaredNumberOfEpochs, 2 );
    BOOST_CHECK_SMALL( fileContents->declaredEpochInterval - 900.0, 1.0E-15 );

    BOOST_REQUIRE_EQUAL( fileContents->satelliteStates.count( "G01" ), 1 );
    const std::map< double, Eigen::VectorXd >& g01States = fileContents->satelliteStates.at( "G01" );
    BOOST_REQUIRE_EQUAL( g01States.size( ), 2 );

    const double expectedTimeFirstEpoch =
            basic_astrodynamics::convertCalendarDateToJulianDaysSinceEpoch< double >(
                    2024, 1, 2, 0, 0, 0.0, basic_astrodynamics::JULIAN_DAY_ON_J2000 ) *
            physical_constants::JULIAN_DAY;
    const double expectedTimeSecondEpoch = expectedTimeFirstEpoch + 15.0 * 60.0 + 30.0;
    BOOST_CHECK_SMALL( fileContents->startEpoch - expectedTimeFirstEpoch, 1.0E-12 );

    auto stateIterator = g01States.begin( );
    BOOST_CHECK_SMALL( stateIterator->first - expectedTimeFirstEpoch, 1.0E-12 );
    BOOST_CHECK_SMALL( stateIterator->second( 0 ) - 12345.0E3, 1.0E-12 );
    BOOST_CHECK_SMALL( stateIterator->second( 1 ) - 23456.0E3, 1.0E-12 );
    BOOST_CHECK_SMALL( stateIterator->second( 2 ) - 34567.0E3, 1.0E-12 );
    BOOST_CHECK_SMALL( stateIterator->second( 3 ) - 0.1, 1.0E-15 );
    BOOST_CHECK_SMALL( stateIterator->second( 4 ) - 0.2, 1.0E-15 );
    BOOST_CHECK_SMALL( stateIterator->second( 5 ) - 0.3, 1.0E-15 );

    ++stateIterator;
    BOOST_CHECK_SMALL( stateIterator->first - expectedTimeSecondEpoch, 1.0E-12 );
    BOOST_CHECK_SMALL( stateIterator->second( 0 ) - 12346.0E3, 1.0E-12 );
    BOOST_CHECK_SMALL( stateIterator->second( 1 ) - 23457.0E3, 1.0E-12 );
    BOOST_CHECK_SMALL( stateIterator->second( 2 ) - 34568.0E3, 1.0E-12 );
    BOOST_CHECK_SMALL( stateIterator->second( 3 ) - 0.4, 1.0E-15 );
    BOOST_CHECK_SMALL( stateIterator->second( 4 ) - 0.5, 1.0E-15 );
    BOOST_CHECK_SMALL( stateIterator->second( 5 ) - 0.6, 1.0E-15 );

    boost::filesystem::remove( temporaryFile );
}

BOOST_AUTO_TEST_SUITE_END( )

}  // namespace unit_tests
}  // namespace tudat
