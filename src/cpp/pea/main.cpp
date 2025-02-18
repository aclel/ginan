
// #pragma GCC optimize ("O0")

#include <sys/time.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <thread>
#include <string>

#ifdef ENABLE_PARALLELISATION
	#include "omp.h"
#endif

using namespace std::literals::chrono_literals;
using std::this_thread::sleep_for;
using std::chrono::system_clock;
using std::chrono::time_point;
using std::make_unique;
using std::make_shared;
using std::string;

#include <boost/log/utility/setup/console.hpp>
#include <boost/system/error_code.hpp>
#include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>



#include "minimumConstraints.hpp"
#include "networkEstimator.hpp"
#include "peaCommitStrings.hpp"
#include "ntripBroadcast.hpp"
#include "rinexNavWrite.hpp"
#include "rinexObsWrite.hpp"
#include "rinexClkWrite.hpp"
#include "algebraTrace.hpp"
#include "rtsSmoothing.hpp"
#include "preprocessor.hpp"
#include "ntripSocket.hpp"
#include "rtcmEncoder.hpp"
#include "sinexParser.hpp"
#include "streamRinex.hpp"
#include "corrections.hpp"
#include "coordinates.hpp"
#include "staticField.hpp"
#include "geomagField.hpp"
#include "binaryStore.hpp"
#include "orbexWrite.hpp"
#include "mongoWrite.hpp"
#include "GNSSambres.hpp"
#include "instrument.hpp"
#include "streamFile.hpp"
#include "streamRtcm.hpp"
#include "ephPrecise.hpp"
#include "acsConfig.hpp"
#include "testUtils.hpp"
#include "streamUbx.hpp"
#include "streamSp3.hpp"
#include "streamSlr.hpp"
#include "oceanTide.hpp"
#include "orbitProp.hpp"
#include "biasSINEX.hpp"
#include "ionoModel.hpp"
#include "ephemeris.hpp"
#include "sp3Write.hpp"
#include "metaData.hpp"
#include "attitude.hpp"
#include "station.hpp"
#include "summary.hpp"
#include "antenna.hpp"
#include "satStat.hpp"
#include "fileLog.hpp"
#include "jpl_eph.hpp"
#include "common.hpp"
#include "orbits.hpp"
#include "gTime.hpp"
#include "trace.hpp"
#include "debug.hpp"
#include "sinex.hpp"
#include "cost.hpp"
#include "enums.h"
#include "ppp.hpp"
#include "gpx.hpp"
#include "slr.hpp"
#include "api.hpp"
#include "trop.h"
#include "vmf3.h"

Navigation				nav		= {};
int						epoch	= 1;
GTime					tsync	= GTime::noTime();
map<int, SatIdentity>	satIdMap;
map<string, Station>	stationMap;

void outputMqtt(KFState& kfState);

bool fileChanged(
	string filename)
{
	bool valid = checkValidFile(filename);
	if (valid == false)
	{
		return false;
	}
	
	auto modifyTime = boost::filesystem::last_write_time(filename);
	
	auto it = acsConfig.configModifyTimeMap.find(filename);
	if (it == acsConfig.configModifyTimeMap.end())
	{
		//the first time this file has been read, 
		//update then return true
		acsConfig.configModifyTimeMap[filename] = modifyTime;
		
		return true;
	}
	
	auto& [dummy, readTime] = *it;
	if (readTime != modifyTime)
	{
		//has a different modification time, update then return true
		readTime = modifyTime;
	
		return true;
	}
	
	//has been read with this time before
	return false;
}

void removeInvalidFiles(
	vector<string>& files)
{
	for (auto it = files.begin(); it != files.end(); )
	{
		auto& filename = *it;
		bool valid = checkValidFile(filename);
		if (valid == false)
		{
			it = files.erase(it);
		}
		else
		{
			it++;
		}
	}
}

void initialiseStation(
	string		id,
	Station&	rec)
{
	if (rec.id.empty() == false)
	{
		//already initialised
		return;
	}
	
	BOOST_LOG_TRIVIAL(info)
	<< "Initialising station " << id;

	Instrument	instrument(__FUNCTION__);
	
	rec.id = id;

	// Read the BLQ file
	bool found = false;
	for (auto& blqfile : acsConfig.blq_files)
	{
		found = readblq(blqfile, id.c_str(), rec.otlDisplacement);

		if (found)
		{
			break;
		}
	}

	if (found == false)
	{
		BOOST_LOG_TRIVIAL(warning)
		<< "Warning: No BLQ for " << id;
	}

	if (acsConfig.process_user)
	{
		rec.pppState.id							= id;
		rec.pppState.max_filter_iter			= acsConfig.pppOpts.max_filter_iter;
		rec.pppState.max_prefit_remv			= acsConfig.pppOpts.max_prefit_remv;
		rec.pppState.inverter					= acsConfig.pppOpts.inverter;
		rec.pppState.sigma_threshold			= acsConfig.pppOpts.sigma_threshold;
		rec.pppState.sigma_check				= acsConfig.pppOpts.sigma_check;
		rec.pppState.w_test						= acsConfig.pppOpts.w_test;
		rec.pppState.chi_square_test			= acsConfig.pppOpts.chi_square_test;
		rec.pppState.chi_square_mode			= acsConfig.pppOpts.chi_square_mode;
		rec.pppState.output_residuals			= acsConfig.output_residuals;
		rec.pppState.outputMongoMeasurements	= acsConfig.localMongo.output_measurements;

		rec.pppState.measRejectCallbacks	.push_back(countSignalErrors);
		rec.pppState.measRejectCallbacks	.push_back(deweightMeas);
		rec.pppState.stateRejectCallbacks	.push_back(rejectByState);
		rec.pppState.stateRejectCallbacks	.push_back(clockGlitchReaction);
	}

	if	( acsConfig.process_rts
		&&acsConfig.pppOpts.rts_lag)
	{
		rec.pppState.rts_lag = acsConfig.pppOpts.rts_lag;
	}
}

/** Create a station object from an input
*/
void addStationData(
	string			stationId,			///< Id of station to add data for
	vector<string>	inputNames,			///< Filename to create station from
	string			inputFormat,		///< Type of data in file
	string			dataType)			///< Type of data
{
	for (auto& inputName : inputNames)
	{
		if (streamDOAMap.find(inputName) != streamDOAMap.end())
		{
			//this stream was already added, dont re-add
			continue;
		}
		
		string mountpoint;
		auto lastSlashPos = inputName.find_last_of('/');
		if (lastSlashPos == string::npos)		{	mountpoint = inputName;								}
		else									{	mountpoint = inputName.substr(lastSlashPos + 1);	};
		
		string id = stationId;
		
		if (id == "<AUTO>")
		{
			id = mountpoint.substr(0,4);
		}
		
		boost::algorithm::to_upper(stationId);

		auto& recOpts = acsConfig.getRecOpts(stationId);

		if (recOpts.exclude)
		{
			return;
		}
		
		string protocol;
		string subInputName;
		auto protocolPos = inputName.find("://");
		if (protocolPos == string::npos)	{	protocol = "file";								subInputName = inputName;							}
		else								{	protocol = inputName.substr(0, protocolPos);	subInputName = inputName.substr(protocolPos + 3);	}
		
		std::unique_ptr<Stream>			stream_ptr;
		std::unique_ptr<Parser>			parser_ptr;
		
		if	( protocol == "file"
			||protocol == "serial")
		{
			if (checkValidFile(subInputName, dataType) == false)
			{
				return;
			}
		}
		
		if		(protocol == "file")		{	stream_ptr = make_unique<FileStream>	(subInputName);	}
		else if (protocol == "serial")		{	stream_ptr = make_unique<SerialStream>	(subInputName);	}
		else if (protocol == "http")		{	stream_ptr = make_unique<NtripStream>	(inputName);	}
		else if (protocol == "https")		{	stream_ptr = make_unique<NtripStream>	(inputName);	}
		else if (protocol == "ntrip")		{	stream_ptr = make_unique<NtripStream>	(inputName);	}
		else
		{
			BOOST_LOG_TRIVIAL(warning)
			<< "Warning: Invalid protocol " << protocol;
		}
		
		if		(inputFormat == "RINEX")	{	parser_ptr = make_unique<RinexParser>	();	}
		else if	(inputFormat == "UBX")		{	parser_ptr = make_unique<UbxParser>		();	}
		else if (inputFormat == "RTCM")		{	parser_ptr = make_unique<RtcmParser>	();	static_cast<RtcmParser*>(parser_ptr.get())->rtcmMountpoint = mountpoint;	}
		else if (inputFormat == "SP3")		{	parser_ptr = make_unique<Sp3Parser>		();	}
		else if (inputFormat == "SINEX")	{	parser_ptr = make_unique<SinexParser>	();	}
		else if (inputFormat == "SLR")		{	parser_ptr = make_unique<SlrParser>		();	}
		else
		{
			BOOST_LOG_TRIVIAL(warning)
			<< "Warning: Invalid inputFormat " << inputFormat;
		}
		
		shared_ptr<StreamParser> streamParser_ptr;
			
		if		(dataType == "OBS")		streamParser_ptr = make_shared<ObsStream>	(std::move(stream_ptr), std::move(parser_ptr));
		else if	(dataType == "PSEUDO")	streamParser_ptr = make_shared<ObsStream>	(std::move(stream_ptr), std::move(parser_ptr));
		else							streamParser_ptr = make_shared<StreamParser>(std::move(stream_ptr), std::move(parser_ptr));
				
		if (dataType == "OBS")
		{
			auto& rec = stationMap[id];
			
			initialiseStation(id, rec);
		}
	
		streamParser_ptr->stream.sourceString = inputName;
		
		streamParserMultimap.insert({id, std::move(streamParser_ptr)});
		
		streamDOAMap[inputName] = false;
	}
}

void reloadInputFiles()
{
	removeInvalidFiles(acsConfig.atx_files);
	for (auto& atxfile : acsConfig.atx_files)
	{
		if (fileChanged(atxfile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading ATX file " << atxfile;

		bool pass = readantexf(atxfile, nav);
		if (pass == false)
		{
			BOOST_LOG_TRIVIAL(error)
			<< "Error: Unable to load ATX from file " << atxfile;

			continue;
		}
	}

	removeInvalidFiles(acsConfig.orb_files);
	removeInvalidFiles(acsConfig.sp3_files);
	if (acsConfig.orb_files.empty() == false)
	{
		bool updated = false;

		/* orbit info from orbit file */
		for (auto& orbfile : acsConfig.orb_files)
		{
			if (fileChanged(orbfile) == false)
			{
				continue;
			}

			updated = true;

			BOOST_LOG_TRIVIAL(info)
			<< "Reading ORB file " << orbfile;

			readorbit(orbfile);
		}

		if (updated)
		{
			orb2sp3(nav);
		}
	}
	else
	{
		for (auto& sp3file : acsConfig.sp3_files)
		{
			if (fileChanged(sp3file) == false)
			{
				continue;
			}

			BOOST_LOG_TRIVIAL(info)
			<< "Loading SP3 file " << sp3file;

			readSp3ToNav(sp3file, &nav, 0);
		}
	}

	removeInvalidFiles(acsConfig.obx_files);
	for (auto& obxfile : acsConfig.obx_files)
	{
		if (fileChanged(obxfile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading OBX file " << obxfile;

		readOrbex(obxfile, nav);
	}

	removeInvalidFiles(acsConfig.nav_files);
	for (auto& navfile : acsConfig.nav_files)
	{
		if (fileChanged(navfile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading NAV file " << navfile;

		auto rinexStream = make_unique<StreamParser>(make_unique<FileStream>(navfile), make_unique<RinexParser>());
		
		rinexStream->parse();
	}

	removeInvalidFiles(acsConfig.erp_files);
	for (auto& erpfile : acsConfig.erp_files)
	{
		if (fileChanged(erpfile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading ERP file " << erpfile;

		readErp(erpfile, nav.erp);
	}

	removeInvalidFiles(acsConfig.clk_files);
	for (auto& clkfile : acsConfig.clk_files)
	{
		if (fileChanged(clkfile) == false)
		{
			continue;
		}

		/* CLK file - RINEX 3 */
		BOOST_LOG_TRIVIAL(info)
		<< "Loading CLK file " << clkfile;

		auto rinexStream = make_unique<StreamParser>(make_unique<FileStream>(clkfile), make_unique<RinexParser>());
		
		rinexStream->parse();
	}

	removeInvalidFiles(acsConfig.dcb_files);
	for (auto& dcbfile : acsConfig.dcb_files)
	{
		if (fileChanged(dcbfile) == false)
		{
			continue;
		}

		/* DCB file */
		BOOST_LOG_TRIVIAL(info)
		<< "Loading DCB file " << dcbfile;

		readdcb(dcbfile);
	}

	removeInvalidFiles(acsConfig.bsx_files);
	for (auto& bsxfile : acsConfig.bsx_files)
	{
		if (fileChanged(bsxfile) == false)
		{
			continue;
		}

		/* BSX file*/
		BOOST_LOG_TRIVIAL(info)
		<< "Loading BSX file " << bsxfile;

		readBiasSinex(bsxfile);
	}

	removeInvalidFiles(acsConfig.ion_files);
	for (auto& ionfile : acsConfig.ion_files)
	{
		if (fileChanged(ionfile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading ION file " << ionfile;

		readTec(ionfile, &nav);
	}

	removeInvalidFiles(acsConfig.igrf_files);
	for (auto& igrffile : acsConfig.igrf_files)
	{
		if (fileChanged(igrffile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading IGRF file " << igrffile;

		readIGRF(igrffile);
	}

	static bool once = true;
	removeInvalidFiles(acsConfig.snx_files);
	for (auto& snxfile : acsConfig.snx_files)
	{
		if (fileChanged(snxfile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading SNX file " <<  snxfile;

		bool fail = readSinex(snxfile, once);
		if (fail)
		{
			BOOST_LOG_TRIVIAL(error)
			<< "Error: Unable to load SINEX file " << snxfile;

			continue;
		}

		once = false;
	}
	
	removeInvalidFiles(acsConfig.vmf_files);
	for (auto& vmffile : acsConfig.vmf_files)
	{
		if (fileChanged(vmffile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading VMF file " << vmffile;

		readvmf3(vmffile, nav.vmf3);
	}
	
	if (acsConfig.model.trop.orography.empty() == false)
	for (auto once : {1})
	{
		if (fileChanged(acsConfig.model.trop.orography) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading ORO from " << acsConfig.model.trop.orography;
		
		readorog(acsConfig.model.trop.orography, nav.vmf3.orography);
	}
	
	if (acsConfig.model.trop.gpt2grid.empty() == false)
	for (auto once : {1})
	{
		if (fileChanged(acsConfig.model.trop.gpt2grid) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading GPT file " << acsConfig.model.trop.gpt2grid;
		
		readgrid(acsConfig.model.trop.gpt2grid, &nav.gptg);
	}
	

	removeInvalidFiles(acsConfig.sid_files); // satellite ID (sp3c code) data
	for (auto& sidfile : acsConfig.sid_files)
	{
		if (fileChanged(sidfile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading Sat ID file " << sidfile;

		readSatId(sidfile);
	}

	removeInvalidFiles(acsConfig.crd_files); // SLR observation data
	for (auto& crdfile : acsConfig.crd_files)
	{
		if (fileChanged(crdfile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading CRD file " << crdfile;

		readCrd(crdfile);
	}
	
	if (acsConfig.output_slr_obs)
	{
		slrObsFiles = outputSortedSlrObs(); // CRD files need to be parsed before sorted .slr_obs files are exported
	}

	removeInvalidFiles(acsConfig.com_files); // centre-of-mass data
	for (auto& comfile : acsConfig.com_files)
	{
		if (fileChanged(comfile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info)
		<< "Loading CoM file " << comfile;

		readCom(comfile);
	}

	removeInvalidFiles(acsConfig.egm_files);
	for (auto& egmfile : acsConfig.egm_files)
	{
		if (fileChanged(egmfile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info) 
		<< "Loading EGM file " << egmfile;

		egm.degMax		= acsConfig.orbitPropagation.degree_max;
		
		egm.readegm(egmfile);
	}	

	removeInvalidFiles(acsConfig.tide_files);
	for (auto& tidefile : acsConfig.tide_files)
	{
		if (fileChanged(tidefile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info) 
		<< "Loading Tide file " << tidefile;

		tide.filename	= tidefile;
		tide.degMax		= acsConfig.orbitPropagation.degree_max;
		
		tide.readocetide();
	}	

	removeInvalidFiles(acsConfig.jpl_files);
	for (auto& jplfile : acsConfig.jpl_files)
	{
		if (fileChanged(jplfile) == false)
		{
			continue;
		}

		BOOST_LOG_TRIVIAL(info) 
		<< "Loading JPL file " << jplfile;
		
		nav.jplEph_ptr = (struct jpl_eph_data*) jpl_init_ephemeris(jplfile.c_str(), nullptr, nullptr);          // a Pointer to The jpl_eph_data Structure
		
		if (jpl_init_error_code())
		{
			BOOST_LOG_TRIVIAL(warning) 
			<< "Warning: JPL file had error code " << jpl_init_error_code();
		}
	}		
	
	for (auto& [id, slrinputs]			: slrObsFiles)					{	addStationData(id,		slrinputs,					"SLR",		"OBS");			}
	for (auto& [id, ubxinputs]			: acsConfig.ubx_inputs)			{	addStationData(id,		ubxinputs,					"UBX",		"OBS");			}
	for (auto& [id, rnxinputs]			: acsConfig.rnx_inputs)			{	addStationData(id,		rnxinputs,					"RINEX",	"OBS");			}	
	for (auto& [id, rtcminputs]			: acsConfig.obs_rtcm_inputs)	{	addStationData(id,		rtcminputs,					"RTCM",		"OBS");			}	
	for (auto& [id, pseudosp3inputs]	: acsConfig.pseudo_sp3_inputs)	{	addStationData(id,		pseudosp3inputs,			"SP3",		"PSEUDO");		}
	for (auto& [id, pseudosnxinputs]	: acsConfig.pseudo_snx_inputs)	{	addStationData(id,		pseudosnxinputs,			"SINEX",	"PSEUDO");		}
																		{	addStationData("Nav",	acsConfig.nav_rtcm_inputs,	"RTCM",		"NAV");			}
}

/** Select 2 clocks for receivers according to the available signals
 */
void setClockCodesForReceiver(
	Trace&		trace,
	Station&	rec)
{
	map<E_Sys,map<E_ObsCode,int>> satsPerCode;
	
	for (auto&	obs				: only<GObs>(rec.obsList))
	for (auto&	[ft, sigList]	: obs.SigsLists)
	for (auto&	sig				: sigList)
	{
		if (rec.recClockCodes.find(obs.Sat.sys) != rec.recClockCodes.end())
			continue;
		
		if	(  sig.L > 0
			&& sig.P > 0)
		{
			satsPerCode[obs.Sat.sys][sig.code]++;
		}
	}
	
	for (auto&	[sys, codeList]	: satsPerCode)
	{
		vector<E_ObsCode> selectedCodes;
		for (auto& code : acsConfig.code_priorities[sys])
		{
			if (codeList.find(code) == codeList.end())
				continue;
			
			if (codeList[code] < 4)		// minimum number for spp
				continue;
			
			if	(  selectedCodes.size() == 1
				&& code2Freq[sys][code] == code2Freq[sys][selectedCodes[0]])
			{
				continue;
			}
			
			selectedCodes.push_back(code);
		
			if (selectedCodes.size() < 2)
				continue;
			
			rec.recClockCodes[sys].first  = selectedCodes[0];
			rec.recClockCodes[sys].second = selectedCodes[1];
			
			trace << std::endl << sys._to_string() << " receiver codes: " <<  selectedCodes[0]._to_string() << " " << selectedCodes[1]._to_string() << std::endl;
			
			break;
		}
	}
}

void configureUploadingStreams()
{
	for (auto& [outLabel, outStreamData] : acsConfig.netOpts.uploadingStreamData)
	{
		auto it = ntripBroadcaster.ntripUploadStreams.find(outLabel);

		// Create stream if it does not already exist.
		if (it == ntripBroadcaster.ntripUploadStreams.end())
		{
			auto outStream_ptr = std::make_shared<NtripUploader>(outStreamData.url);
			auto& outStream = *outStream_ptr.get();
			ntripBroadcaster.ntripUploadStreams[outLabel] = std::move(outStream_ptr);

			it = ntripBroadcaster.ntripUploadStreams.find(outLabel);
		}

		auto& [label, outStream_ptr]	= *it;
		auto& outStream					= *outStream_ptr;
		
		outStream.streamConfig.rtcmMsgOptsMap		= outStreamData.rtcmMsgOptsMap;
		outStream.streamConfig.itrf_datum 			= outStreamData.itrf_datum;
		outStream.streamConfig.provider_id 			= outStreamData.provider_id;
		outStream.streamConfig.solution_id 			= outStreamData.solution_id;
		outStream.streamConfig.master_iod 			= outStreamData.master_iod;
	}

	for (auto it = ntripBroadcaster.ntripUploadStreams.begin(); it != ntripBroadcaster.ntripUploadStreams.end();)
	{
		if (acsConfig.netOpts.uploadingStreamData.find(it->first) == acsConfig.netOpts.uploadingStreamData.end())
		{
			auto& [label, outStream_ptr]	= *it;
			auto& outStream					= *outStream_ptr;
			outStream.disconnect();
			it = ntripBroadcaster.ntripUploadStreams.erase(it);
		}
		else
		{
			it++;
		}
	}
}

bool createNewTraceFile(
	const string				id,
	boost::posix_time::ptime	logptime,
	string  					new_path_trace,
	string& 					old_path_trace,
	bool						outputHeader = false,
	bool						outputConfig = false)
{	
	replaceString(new_path_trace, "<STATION>", id);
	replaceTimes (new_path_trace, logptime);

	// Create the trace file if its a new filename, otherwise, keep the old one
	if	( new_path_trace == old_path_trace
		||new_path_trace.empty())
	{
		//the filename is the same, keep using the old ones
		return false;
	}

	old_path_trace = new_path_trace;

	BOOST_LOG_TRIVIAL(debug)
	<< "Creating new file for " << id << " at " << old_path_trace;

	std::ofstream trace(old_path_trace);
	if (!trace)
	{
		BOOST_LOG_TRIVIAL(error)
		<< "Error: Could not create file for " << id << " at " << old_path_trace;

		return false;
	}

	// Trace file head
	if (outputHeader)
	{
		trace << "station    : " << id << std::endl;
		trace << "start_epoch: " << acsConfig.start_epoch			<< std::endl;
		trace << "end_epoch  : " << acsConfig.end_epoch				<< std::endl;
		trace << "trace_level: " << acsConfig.trace_level			<< std::endl;
		trace << "pea_version: " << ginanCommitVersion()			<< std::endl;
// 		trace << "rts_lag    : " << acsConfig.pppOpts.rts_lag		<< std::endl;
	}

	if (outputConfig)
	{
		dumpConfig(trace);
	}
	
	return true;
}

/** Returns (posix) for current epoch
*/
boost::posix_time::ptime currentLogptime()
{
	PTime logtime = tsync.floorTime(acsConfig.rotate_period);
	
	boost::posix_time::ptime	logptime	= boost::posix_time::from_time_t((time_t)logtime.bigTime);
	
	if ((GTime)logtime == GTime::noTime())
	{
		logptime = boost::posix_time::not_a_date_time;
	}
	return logptime;
}

/** Create directories if required
 */
void createDirectories(
	boost::posix_time::ptime	logptime)
{
	// Ensure the output directories exist
	for (auto directory : {
								acsConfig.sp3_directory,
								acsConfig.erp_directory,
								acsConfig.gpx_directory,
								acsConfig.log_directory,
								acsConfig.cost_directory,
								acsConfig.test_directory,
								acsConfig.ionex_directory,
								acsConfig.orbex_directory,
								acsConfig.sinex_directory,
								acsConfig.trace_directory,
								acsConfig.clocks_directory,
								acsConfig.slr_obs_directory,
								acsConfig.ionstec_directory,
								acsConfig.ppp_sol_directory,
								acsConfig.rtcm_nav_directory,
								acsConfig.rtcm_obs_directory,
								acsConfig.orbit_ics_directory,
								acsConfig.rinex_obs_directory,
								acsConfig.rinex_nav_directory,
								acsConfig.trop_sinex_directory,
								acsConfig.bias_sinex_directory,
								acsConfig.persistance_directory,
								acsConfig.pppOpts.rts_directory,
								acsConfig.decoded_rtcm_json_directory,
								acsConfig.encoded_rtcm_json_directory,
								acsConfig.network_statistics_json_directory
							})
	{
		replaceTimes(directory, logptime);
		
		if (directory == "./")	continue;
		if (directory.empty())	continue;
		
		bool created = boost::filesystem::create_directories(directory);
		{
			if (created)
			{
				created = false;
			}
		}
	}
}

map<string, string> fileNames;

/** Create new empty trace files only when required when the filename is changed
 */
void createTracefiles(
	map<string, Station>&	stationMap,
	Network&				net)
{
	boost::posix_time::ptime logptime = currentLogptime();
	createDirectories(logptime);
	
	for (auto rts : {false, true})
	{
		if	(	rts 
			&&(	acsConfig.process_rts		== false
			  ||acsConfig.pppOpts.rts_lag	== 0))
		{
			continue;
		}

		
		string suff		= "";
		string metaSuff	= "";
		
		if (rts)
		{
			suff		= acsConfig.pppOpts.rts_smoothed_suffix;
			metaSuff	= SMOOTHED_SUFFIX;
			
			if	( acsConfig.process_network
				||acsConfig.process_ppp)
			{
				bool newTraceFile = createNewTraceFile(net.id,		boost::posix_time::not_a_date_time,	acsConfig.pppOpts.rts_filename,		net.kfState.rts_basename);
			
				if (newTraceFile)
				{
	// 				std::cout << std::endl << "new trace file";
					std::remove((net.kfState.rts_basename					).c_str());
					std::remove((net.kfState.rts_basename + FORWARD_SUFFIX	).c_str());
					std::remove((net.kfState.rts_basename + BACKWARD_SUFFIX	).c_str());
				}
			}
			
			if (acsConfig.process_user)
			for (auto& [id, rec] : stationMap)
			{
				bool newTraceFile = createNewTraceFile(id,			boost::posix_time::not_a_date_time,	acsConfig.pppOpts.rts_filename,		rec.pppState.rts_basename);
				
				if (newTraceFile)
				{
// 					std::cout << std::endl << "new trace file";
					std::remove((rec.pppState.rts_basename					).c_str());
					std::remove((rec.pppState.rts_basename + FORWARD_SUFFIX	).c_str());
					std::remove((rec.pppState.rts_basename + BACKWARD_SUFFIX).c_str());
				}
			}
		}
		
		bool newTraceFile = false;
		
		for (auto& [Sat, satNav] : nav.satNavMap)
		{
			if	(  acsConfig.output_satellite_trace
				&& suff.empty())
			{
				newTraceFile |= createNewTraceFile(Sat,			logptime,	acsConfig.satellite_trace_filename		+ suff,	satNav.traceFilename,												true,	acsConfig.output_config);
			}	
		}
		
		for (auto& [id, rec] : stationMap)
		{	
			if (acsConfig.output_station_trace)
			{
				newTraceFile |= createNewTraceFile(id,				logptime,	acsConfig.station_trace_filename	+ suff,	rec.pppState.metaDataMap[TRACE_FILENAME_STR			+ metaSuff],	true,	acsConfig.output_config);
				
				if (suff.empty())
				{
					rec.traceFilename = rec.pppState.metaDataMap[TRACE_FILENAME_STR];
				}
			}
			
			if	( acsConfig.output_trop_sinex
				&&acsConfig.process_user)
			{
				if (acsConfig.process_user)
				{
					newTraceFile |= createNewTraceFile(id,			logptime,	acsConfig.trop_sinex_filename		+ suff,	rec.pppState.metaDataMap[TROP_FILENAME_STR			+ metaSuff]);
				}
				
				if (acsConfig.process_ppp)
				{
					newTraceFile |= createNewTraceFile(id,			logptime,	acsConfig.trop_sinex_filename		+ suff,	net.kfState	.metaDataMap[TROP_FILENAME_STR			+ metaSuff]);
				}
			}

			if	(acsConfig.output_cost)
			{
				if (acsConfig.process_user)
				{
					newTraceFile |= createNewTraceFile(id,			logptime,	acsConfig.cost_filename				+ suff,	rec.pppState.metaDataMap[COST_FILENAME_STR	+ id	+ metaSuff]);
				}
				
				if (acsConfig.process_ppp)
				{
					newTraceFile |= createNewTraceFile(id,			logptime,	acsConfig.cost_filename				+ suff,	net.kfState	.metaDataMap[COST_FILENAME_STR	+ id	+ metaSuff]);
				}
			}
			
			if (acsConfig.output_ppp_sol)
			{
				if (acsConfig.process_user)
				{
					newTraceFile |= createNewTraceFile(id,			logptime,	acsConfig.ppp_sol_filename			+ suff,	rec.pppState.metaDataMap[SOL_FILENAME_STR	+ id	+ metaSuff],	true,	acsConfig.output_config);
				}
				
				if (acsConfig.process_ppp)
				{
					newTraceFile |= createNewTraceFile(id,			logptime,	acsConfig.ppp_sol_filename			+ suff,	net.kfState	.metaDataMap[SOL_FILENAME_STR	+ id	+ metaSuff],	true,	acsConfig.output_config);
				}
			}
			
			if (acsConfig.output_gpx)
			{
				if (acsConfig.process_user)
				{
					newTraceFile |= createNewTraceFile(id,			logptime,	acsConfig.gpx_filename				+ suff,	rec.pppState.metaDataMap[GPX_FILENAME_STR	+ id	+ metaSuff]);
				}
				
				if	( acsConfig.process_ppp
					||acsConfig.process_network)
				{
					newTraceFile |= createNewTraceFile(id,			logptime,	acsConfig.gpx_filename				+ suff,	net.kfState	.metaDataMap[GPX_FILENAME_STR	+ id	+ metaSuff]);
				}
			}
			
					
			if	(  rts 
				&& newTraceFile
				&& rec.pppState.rts_basename.empty() == false)
			{
				spitFilterToFile(rec.pppState.metaDataMap,	E_SerialObject::METADATA, rec.pppState.rts_basename + FORWARD_SUFFIX); 
			}
		}
		
		if (acsConfig.output_network_trace)
		{
			newTraceFile |= createNewTraceFile(net.id,	logptime,	acsConfig.network_trace_filename	+ suff,	net.kfState.metaDataMap[TRACE_FILENAME_STR		+ metaSuff],	true,	acsConfig.output_config);
			
			if (suff.empty())
			{
				net.traceFilename = net.kfState.metaDataMap[TRACE_FILENAME_STR];
			}
		}

		if (acsConfig.output_ionex)
		{
			newTraceFile |= createNewTraceFile("",		logptime,	acsConfig.ionex_filename			+ suff,	net.kfState.metaDataMap[IONEX_FILENAME_STR		+ metaSuff]);
			newTraceFile |= createNewTraceFile("",		logptime,	acsConfig.ionex_filename			+ suff,	iono_KFState.metaDataMap[IONEX_FILENAME_STR		+ metaSuff]);
		
			newTraceFile |= createNewTraceFile("IONO",	logptime,	acsConfig.network_trace_filename	+ suff,	iono_KFState.metaDataMap[TRACE_FILENAME_STR		+ metaSuff],	true,	acsConfig.output_config);
		}

		if (acsConfig.output_ionstec)
		{
			newTraceFile |= createNewTraceFile("",		logptime,	acsConfig.ionstec_filename			+ suff,	net.kfState.metaDataMap[IONSTEC_FILENAME_STR	+ metaSuff]);
		}
	
		if	( ( acsConfig.output_trop_sinex)
			&&( acsConfig.process_ppp
			  ||acsConfig.process_network))
		{
			newTraceFile |= createNewTraceFile(net.id,	logptime,	acsConfig.trop_sinex_filename		+ suff,	net.kfState.metaDataMap[TROP_FILENAME_STR		+ metaSuff]);
		}

		if (acsConfig.output_bias_sinex)
		{
			newTraceFile |= createNewTraceFile(net.id, 	logptime,	acsConfig.bias_sinex_filename		+ suff,	net.kfState.metaDataMap[BSX_FILENAME_STR		+ metaSuff]);
			newTraceFile |= createNewTraceFile(net.id, 	logptime,	acsConfig.bias_sinex_filename		+ suff,	iono_KFState.metaDataMap[BSX_FILENAME_STR		+ metaSuff]);
		}	
		
		if (acsConfig.output_erp)
		{	
			newTraceFile |= createNewTraceFile(net.id,	logptime,	acsConfig.erp_filename				+ suff,	net.kfState.metaDataMap[ERP_FILENAME_STR		+ metaSuff]);
		}
		
		if (acsConfig.output_clocks)
		{
			auto singleFilenameMap	= getSysOutputFilenames(acsConfig.clocks_filename,	tsync, false);
			auto filenameMap		= getSysOutputFilenames(acsConfig.clocks_filename,	tsync);
			for (auto& [filename, dummy] : filenameMap)
			{
				newTraceFile |= createNewTraceFile(net.id,	logptime,	filename + suff,				fileNames[filename + metaSuff]);
			}
		
			net.kfState.metaDataMap[CLK_FILENAME_STR	+ metaSuff] = singleFilenameMap.begin()->first + suff;
		}	

		if (acsConfig.output_sp3)
		{
			auto singleFilenameMap	= getSysOutputFilenames(acsConfig.sp3_filename,	tsync, false);
			auto filenameMap		= getSysOutputFilenames(acsConfig.sp3_filename,	tsync);
			for (auto& [filename, dummy] : filenameMap)
			{
				newTraceFile |= createNewTraceFile(net.id,	logptime,	filename + suff,				fileNames[filename + metaSuff]);
			}

			net.kfState.metaDataMap[SP3_FILENAME_STR	+ metaSuff] = singleFilenameMap.begin()->first + suff;
		}

		if (acsConfig.output_orbex)
		{
			auto singleFilenameMap	= getSysOutputFilenames(acsConfig.orbex_filename,	tsync, false);
			auto filenameMap		= getSysOutputFilenames(acsConfig.orbex_filename,	tsync);
			for (auto& [filename, dummy] : filenameMap)
			{
				newTraceFile |= createNewTraceFile(net.id,	logptime,	filename + suff,				fileNames[filename + metaSuff],					false,	false);
			}

			net.kfState.metaDataMap[ORBEX_FILENAME_STR	+ metaSuff] = singleFilenameMap.begin()->first + suff;
		}
		
		if	(  rts 
			&& newTraceFile)
		{
			spitFilterToFile(net.kfState.metaDataMap,	E_SerialObject::METADATA, net.kfState.rts_basename + FORWARD_SUFFIX); 
		}
	}
	
	if (acsConfig.output_log)
	{
		createNewTraceFile("",			logptime,	acsConfig.log_filename,								FileLog::path_log);
	}
	
	if (acsConfig.output_ntrip_log)
	{
		for (auto& [id, stream_ptr] : ntripBroadcaster.ntripUploadStreams)
		{
			auto& stream = *stream_ptr;
			
			createNewTraceFile(id,			logptime,	acsConfig.ntrip_log_filename,					stream.networkTraceFilename);
		}
		
		for (auto& [id, streamParser_ptr] : streamParserMultimap)
		try
		{			
			auto& ntripStream = dynamic_cast<NtripStream&>(streamParser_ptr->stream);
			
			createNewTraceFile(id,			logptime,	acsConfig.ntrip_log_filename,					ntripStream.networkTraceFilename);
		}
		catch(...){}
	}

	if (acsConfig.output_rinex_obs)
	for (auto& [id, rec] : stationMap)
	{
		auto filenameMap = getSysOutputFilenames(acsConfig.rinex_obs_filename,	tsync, true, id);
		for (auto& [filename, dummy] : filenameMap)
		{
			createNewTraceFile(id,		logptime,	filename,											fileNames[filename]);
		}
	}
	
	if (acsConfig.output_rinex_nav)
	{
		auto filenameMap = getSysOutputFilenames(acsConfig.rinex_nav_filename,	tsync);
		for (auto& [filename, dummy] : filenameMap)
		{
			createNewTraceFile("Navs",	logptime,	filename,											fileNames[filename]);
		}
	}

	for (auto& [id, streamParser_ptr] : streamParserMultimap)
	try
	{
		auto& rtcmParser = dynamic_cast<RtcmParser&>(streamParser_ptr->parser);
		
		if (acsConfig.output_decoded_rtcm_json)
		{
			string filename = acsConfig.decoded_rtcm_json_filename;
			
			replaceString(filename, "<STREAM>",		rtcmParser.rtcmMountpoint);
			
			createNewTraceFile(id, 		logptime,	filename,	rtcmParser.rtcmTraceFilename);
		}
		
		for (auto nav : {false, true})
		{
			bool isNav = true;
			try
			{
				auto& obsStream = dynamic_cast<ObsStream&>(*streamParser_ptr);
				
				isNav = false;
			}
			catch(...){}
			
			if	( (acsConfig.record_rtcm_nav && isNav == true	&& nav == true)
				||(acsConfig.record_rtcm_obs && isNav == false	&& nav == false))
			{
				string filename;
				
				if (nav)	filename = acsConfig.rtcm_nav_filename;
				else		filename = acsConfig.rtcm_obs_filename; 
				
				replaceString(filename, "<STREAM>",		rtcmParser.rtcmMountpoint);
				
				createNewTraceFile(id, 		logptime,	filename,	rtcmParser.rtcm_filename);
			}
		}
	}
	catch(...){}
	
	for (auto& [id, streamParser_ptr] : streamParserMultimap)
	try
	{
		auto& ubxParser = dynamic_cast<UbxParser&>(streamParser_ptr->parser);
		
		if (acsConfig.record_raw_ubx)
		{
			string filename = acsConfig.raw_ubx_filename;
			
			createNewTraceFile(id, 		logptime,	filename,	ubxParser.raw_ubx_filename);
		}
	}
	catch(...){}
}

void avoidCollisions(
	StationMap&		stationMap)
{
	for (auto& [id, rec] : stationMap)
	{
		auto trace = getTraceFile(rec);
	}
	
	for (auto& [id, rec] : stationMap)
	{
		//create sinex estimate maps
		theSinex.map_estimates_primary	[id];
		theSinex.map_estimates			[id];
	}
}

/** Perform operations for each station
 * This function occurs in parallel with other stations - ensure that any operations on global maps do not create new entries, as that will destroy the map for other processes.
 * Variables within the rec object are ok to use, but be aware that pointers from the within the receiver often point to global variables.
 * Prepare global maps by accessing the desired elements before calling this function.
 */
void mainOncePerEpochPerStation(
	Station&	rec,
	Network&	net,
	bool&		emptyEpoch)
{
	Instrument instrument(__FUNCTION__);
	
	auto trace = getTraceFile(rec);
	
	if (acsConfig.process_spp)
	{
		sppos(trace, rec.obsList, rec.sol, rec.id);
		
		//recalculate variances now that elevations are known due to satellite postions calculation above
		obsVariances(rec.obsList);
	}
	
	if	(  rec.ready == false
		|| rec.invalid)
	{
		return;
	}
	
	sinexPerEpochPerStation(tsync, rec);
	
	bool sppUsed;
	selectAprioriSource(rec, sppUsed);
	
	setClockCodesForReceiver(trace, rec);
	
	if	( sppUsed
		&&acsConfig.require_apriori_positions)
	{
		trace << std::endl			<< "Station " << rec.id << " rejected due to lack of apriori position";
		BOOST_LOG_TRIVIAL(warning)	<< "Station " << rec.id << " rejected due to lack of apriori position";
		
		rec.invalid = true;
		return;
	}
	if	( rec.antennaId.empty()
		&&acsConfig.require_antenna_details)
	{
		trace << std::endl			<< "Station " << rec.id << " rejected due to lack of antenna details";
		BOOST_LOG_TRIVIAL(warning)	<< "Station " << rec.id << " rejected due to lack of antenna details";
		
		rec.invalid = true;
		return;
	}
	
	emptyEpoch = false;
	
	
	BOOST_LOG_TRIVIAL(trace)
	<< "Read " << rec.obsList.size()
	<< " observations for station " << rec.id;

	//calculate statistics
	{
		Instrument instrument("Statistics");
		if ((GTime) rec.firstEpoch	== GTime::noTime())		{	rec.firstEpoch	= rec.obsList.front()->time;		}
																rec.lastEpoch	= rec.obsList.front()->time;
		rec.epochCount++;
		rec.obsCount += rec.obsList.size();

		for (auto& obs				: only<GObs>(rec.obsList))
		for (auto& [ft, sigList]	: obs.SigsLists)
		for (auto& sig				: sigList)
		{
			rec.codeCount[sig.code]++;
		}

		for (auto& obs				: only<GObs>(rec.obsList))
		{
			rec.satCount[obs.Sat]++;
		}
	}

	if (acsConfig.process_ionosphere)
	{
		if (rec.aprioriPos.isZero() == false)
			update_receivr_measr(trace, rec);
	}

	auto& recOpts = acsConfig.getRecOpts(rec.id);
	
	rec.antBoresight	= recOpts.antenna_boresight;
	rec.antAzimuth		= recOpts.antenna_azimuth;
	
	recAtt(rec, tsync, recOpts.rec_attitude.sources);

	if (acsConfig.process_user)
	{
		Instrument instrument("ppp");
		pppos(trace, rec.obsList, rec);
		
		if (rec.sol.status != +E_Solution::NONE)
		{
			int nfixed = 0;
			
			rec.pppState.outputStates(trace);
			pppoutstat(trace, rec.pppState, rec.id);
			
			if (acsConfig.ambrOpts.mode != +E_ARmode::OFF)
			{
				TempDisabler td(rec.pppState.outputMongoMeasurements);
				
				nfixed = enduserAmbigResl(trace, rec.obsList, rec.pppState, rec.snx.pos, rec.sol.dop[2], rec.pppState.metaDataMap[SOL_FILENAME_STR + rec.id]);
			}
			
			trace << std::endl << "Solution with " << nfixed << " ambigities resolved";
			
			if (acsConfig.output_ppp_sol)
			{
				outputPPPSolution(rec.pppState.metaDataMap[SOL_FILENAME_STR + rec.id], rec);
			}
		}
	}
	
	if (acsConfig.process_ppp)
	{
		pppoutstat(trace, net.kfState, rec.id);
	}

	if (acsConfig.process_network)
	for (auto once : {1})
	{
		// If there is no antenna information skip processing this station
		if (rec.antennaType.empty())
		{
			BOOST_LOG_TRIVIAL(warning)
			<< "Warning: \tNo Antenna Information for " << rec.id
			<< " skipping this station";

			rec.invalid = true;
			
			break;
		}

		/* observed minus computed for each satellites */
		pppomc(trace, rec.obsList, nav.gptg, rec, nav.vmf3);
	}

	if	(  acsConfig.process_rts
		&& acsConfig.pppOpts.rts_lag > 0)
	{
		KFState rts = RTS_Process(rec.pppState);
	}

	if (acsConfig.process_user)
	{
		mongoStates(rec.pppState);
		storeStates(rec.pppState);
	}

	if (acsConfig.output_rinex_obs)
	{
		writeRinexObs(rec.id, rec.snx, tsync, rec.obsList, acsConfig.rinex_obs_version);
	}

	if (acsConfig.output_gpx)
	{
		if (acsConfig.process_user)
		{
			writeGPX(rec.pppState.metaDataMap[GPX_FILENAME_STR + rec.id],	rec.id, rec.pppState);
		}
		
		if	( acsConfig.process_ppp
			||acsConfig.process_network)
		{
			writeGPX(net.kfState.metaDataMap[GPX_FILENAME_STR + rec.id],	rec.id, net.kfState);
		}
	}
}

void outputPredictedStates(
	Trace&			trace,
	Network&		net,
	MongoOptions&	mongoOptions)
{
	if (mongoOptions.predict_states == false)
	{
		return;
	}
		
	tuple<double, double>	forward = {+1, mongoOptions.forward_prediction_duration};
	tuple<double, double>	reverse = {-1, mongoOptions.reverse_prediction_duration};

	for (auto& duo : {forward, reverse})
	{
		auto& [sign, duration] = duo;
		
		if (duration < 0)
		{
			continue;
		}
		
		Orbits orbits = prepareOrbits(trace, net.kfState);
		
		if (orbits.empty())
		{
			continue;
		}
		
		GTime orbitsTime	= tsync;
		GTime stopTime		= tsync + sign * duration;
		
		for (GTime time = tsync; sign * (time - stopTime).to_double() <= 0; time += sign * mongoOptions.prediction_interval)
		{
			OrbitIntegrator integrator;
			integrator.timeInit				= orbitsTime;
			integrator.propagationOptions	= acsConfig.orbitPropagation;

			double tgap = (time - orbitsTime).to_double();
			
			integrateOrbits(integrator, orbits, tgap, acsConfig.orbitPropagation.integrator_time_step);
			
			orbitsTime = time;
			
			outputMongoPredictions(trace, orbits, time, mongoOptions);
	
			if	(acsConfig.output_orbit_ics)
			for (auto& orbit : orbits)
			{
				outputOrbitConfig(orbit.subState, "_prop");
			}
		}
	}
}

void mainPerEpochPostProcessingAndOutputs(
	Network&		net,
	StationMap&		stationMap)
{
	TestStack	ts(__FUNCTION__);
	Instrument	instrument(__FUNCTION__);
	
	auto netTrace = getTraceFile(net);

	if (acsConfig.process_user)
	{
		if (acsConfig.output_trop_sinex)
		{
			vector<KFState*> kfStatePointers;
			for (auto& [key, rec] : stationMap)
			{
				kfStatePointers.push_back(&rec.pppState);
			}
			KFState mergedKfState = mergeFilters(kfStatePointers, true);
			for (auto& [id, rec] : stationMap)
			{
				outputTropSinex(rec.pppState.metaDataMap[TROP_FILENAME_STR], tsync, mergedKfState, id);
			}
		}
		
		if (acsConfig.output_clocks)
		{
			outputClocks(acsConfig.clocks_filename, acsConfig.clocks_receiver_sources, acsConfig.clocks_satellite_sources, tsync, net.kfState, &stationMap);
		}
	}
	
	if	( acsConfig.process_ppp
		||acsConfig.process_network)
	{
		mongoStates(net.kfState);
		
		if (acsConfig.output_trop_sinex)
		{
			outputTropSinex(net.kfState.metaDataMap[TROP_FILENAME_STR],				tsync, net.kfState, "MIX");
		}
	}
	
	if (acsConfig.output_cost)
	for (auto& [id, rec] : stationMap)
	{
		if (acsConfig.process_user)
		{
			outputCost(rec.pppState.metaDataMap[COST_FILENAME_STR + id],	rec,	tsync, rec.pppState);
		}
		
		if (acsConfig.process_ppp)
		{
			outputCost(net.kfState.metaDataMap[COST_FILENAME_STR + id],		rec,	tsync, net.kfState);
		}
	}
	
	
	KFState tempAugmentedKF = net.kfState;
	
	tempAugmentedKF.rts_basename = "";
	
	if (acsConfig.process_network)
	{
		tempAugmentedKF.outputStates(netTrace);
		
		KFState KF_ARcopy = net.kfState;
		
		KF_ARcopy.outputMongoMeasurements = false;
		
		if (acsConfig.ambrOpts.mode != +E_ARmode::OFF)
		{
			networkAmbigResl(netTrace, stationMap, KF_ARcopy);
			
			if (ARsol_ready())
			{
				KF_ARcopy.outputStates(netTrace, "/AR");
			}

			mongoStates(KF_ARcopy, "_AR");
		}
		
		if (acsConfig.output_clocks)
		{
			if	( acsConfig.output_ar_clocks == false
				|| ARsol_ready() == false)
			{
				outputClocks(acsConfig.clocks_filename, acsConfig.clocks_receiver_sources, acsConfig.clocks_satellite_sources, tsync, tempAugmentedKF,	&stationMap);
			}
			else
			{
				outputClocks(acsConfig.clocks_filename, acsConfig.clocks_receiver_sources, acsConfig.clocks_satellite_sources, tsync, KF_ARcopy,		&stationMap);
			}
		}
		
		if	(  acsConfig.process_minimum_constraints
			&& acsConfig.minCOpts.once_per_epoch)
		{
			BOOST_LOG_TRIVIAL(info)
			<< std::endl
			<< "---------------PROCESSING NETWORK WITH MINIMUM CONSTRAINTS ------------- " << std::endl;
			
			for (auto& [id, rec] : stationMap)
			{
				rec.minconApriori = rec.aprioriPos;
			}
			
			mincon(netTrace, tempAugmentedKF);
	
			mongoStates(tempAugmentedKF, "_mincon");
		}
		
		if (acsConfig.output_erp)
		{
			writeErpFromNetwork(net.kfState.metaDataMap[ERP_FILENAME_STR], net.kfState);
		}

		if	(  acsConfig.process_rts
			&& acsConfig.pppOpts.rts_lag > 0)
		{
			RTS_Process(net.kfState,	false, &stationMap);
			RTS_Process(KF_ARcopy,		false, &stationMap);

			if (ARsol_ready())
			{
				outputClocks(acsConfig.clocks_filename, acsConfig.clocks_receiver_sources, acsConfig.clocks_satellite_sources, tsync, KF_ARcopy, &stationMap);
			}
		}
	}
	
	if (acsConfig.process_ppp)
	{
		KFState KF_ARcopy;
		
		/* select ambiguity resolved KF */
		if (acsConfig.ambrOpts.mode != +E_ARmode::OFF)
		{
			if (copyFixedKF(KF_ARcopy))
			{
				tempAugmentedKF = KF_ARcopy;
			}
			
			mongoStates(tempAugmentedKF, "_AR");
		}	
			
		
		
		if	(  acsConfig.process_minimum_constraints
			&& acsConfig.minCOpts.once_per_epoch)
		{
			BOOST_LOG_TRIVIAL(info)
			<< std::endl
			<< "---------------PROCESSING NETWORK WITH MINIMUM CONSTRAINTS ------------- " << std::endl;
			
			for (auto& [id, rec] : stationMap)
			{
				rec.minconApriori = rec.aprioriPos;
			}
			
			mincon(netTrace, tempAugmentedKF);
	
			mongoStates(tempAugmentedKF, "_mincon");
		}
		
		if (acsConfig.output_erp)
		{
			writeErpFromNetwork(net.kfState.metaDataMap[ERP_FILENAME_STR], net.kfState);
		}
		
		if (acsConfig.output_clocks)
		{
			outputClocks(acsConfig.clocks_filename, acsConfig.clocks_receiver_sources, acsConfig.clocks_satellite_sources, tsync, tempAugmentedKF, &stationMap);
		}
		
		if	(  acsConfig.process_rts
			&& acsConfig.pppOpts.rts_lag > 0)
		{
			RTS_Process(net.kfState,	false, &stationMap);
		}

		outputStatistics(netTrace, net.kfState.statisticsMap, net.kfState.statisticsMapSum);
	}
	
	{
		if	(  ARsol_ready() 
			&& acsConfig.output_ar_clocks)
		{
			KFState KF_ARcopy = retrieve_last_ARcopy();
			prepareSsrStates(netTrace, KF_ARcopy,		tsync);
		}
		else
		{
			prepareSsrStates(netTrace, tempAugmentedKF,	tsync);
		}
	}
	
	if (acsConfig.output_sp3)
	{
		outputSp3(acsConfig.sp3_filename, tsync, acsConfig.sp3_orbit_sources, acsConfig.sp3_clock_sources, &tempAugmentedKF);
	}
	
	if (acsConfig.output_orbex)
	{
		outputOrbex(acsConfig.orbex_filename, tsync, acsConfig.orbex_orbit_sources, acsConfig.orbex_clock_sources, acsConfig.orbex_attitude_sources, &net.kfState);
	}
	
	if (acsConfig.output_ionex)
	{
		if (acsConfig.process_ionosphere)			ionexFileWrite(iono_KFState.metaDataMap	[IONEX_FILENAME_STR], tsync, iono_KFState);			
		else										ionexFileWrite(net.kfState.metaDataMap	[IONEX_FILENAME_STR], tsync, net.kfState);					
	}

	if (acsConfig.output_persistance)
	{
		outputPersistanceNav();
	}
	
	outputApriori		(stationMap);
	outputDeltaClocks	(stationMap);
	
	if (acsConfig.output_bias_sinex)
	{
		writeBiasSinex(netTrace, tsync, net.kfState.metaDataMap[BSX_FILENAME_STR], stationMap);
	}
	
	if (acsConfig.output_orbit_ics)
	{
		outputOrbitConfig(net.kfState);
	}
	
	outputPredictedStates(netTrace, net, acsConfig.remoteMongo);
	
	mongoCull(tsync);
}

void mainOncePerEpochPerSatellite(
	Trace&	trace,
	GTime	time,
	SatSys	Sat)
{
	auto& satNav 	= nav.satNavMap[Sat];
	auto& satOpts	= acsConfig.getSatOpts(Sat);
	
	if (satOpts.exclude)
	{
		return;
	}
	
	//get svn and block type if possible
	if (Sat.svn().empty())
	{
		auto it = nav.svnMap[Sat].lower_bound(time);
		if (it == nav.svnMap[Sat].end())
		{
			BOOST_LOG_TRIVIAL(warning) << "Warning: SVN not found for " << Sat.id();
			
			Sat.setSvn("UNKNOWN");
		}
		else
		{
			Sat.setSvn(it->second);
		}
		
		//reinitialise the options with the updated values
		satOpts._initialised = false;
	}
	
	if (Sat.blockType().empty())
	{
		auto it = nav.blocktypeMap.find(Sat.svn());
		if (it == nav.blocktypeMap.end())
		{
			BOOST_LOG_TRIVIAL(warning) << "Warning: Block type not found for " << Sat.id() << ", attitude modelling etc may be affected, check sinex file";
			
			Sat.setBlockType("UNKNOWN");
		}
		else
		{
			Sat.setBlockType(it->second);
		}		
		
		//reinitialise the options with the updated values
		satOpts._initialised = false;
	}
	
	satOpts = acsConfig.getSatOpts(Sat);
	
	GObs obs;
	obs.Sat			= Sat;
	obs.time		= time;
	obs.satNav_ptr	= &satNav;	
	
	bool pass =	satpos(nullStream, time, time, obs, satOpts.sat_pos.ephemeris_sources, E_OffsetType::COM, nav);
	if (pass == false)
	{
		BOOST_LOG_TRIVIAL(warning) << "Warning: No sat pos found for " << obs.Sat.id() << ".";
	}
	
	ERPValues erpv = getErp(nav.erp, tsync);
	
	FrameSwapper frameSwapper(time, erpv);
	
	obs.rSatEci0 = frameSwapper(obs.rSat);
	
	satNav.aprioriPos	= obs.rSatEci0;
	satNav.antBoresight	= satOpts.antenna_boresight;
	satNav.antAzimuth	= satOpts.antenna_azimuth;
	
	updateSatAtts(obs);
}


void mainOncePerEpoch(
	Network&		net,
	StationMap&		stationMap,
	GTime			time)
{
	//load any changes from the config
	bool newConfig = acsConfig.parse();
	
	avoidCollisions(stationMap);
	
	//reload any new or modified files
	reloadInputFiles();

	addDefaultBiasSinex();
	
	createTracefiles(stationMap, net);
	
	//make any changes to streams.
	if (newConfig)
	{
		configureUploadingStreams();
	}
	
	auto netTrace = getTraceFile(net);
	
	//initialise mongo if not already done
	mongoooo();

	//try to get svns & block types of all used satellites
	for (auto& [Sat, satNav] : nav.satNavMap)
	{
		if (acsConfig.process_sys[Sat.sys] == false)
			continue;
	
		mainOncePerEpochPerSatellite(netTrace, time, Sat);
	}

	//do per-station pre processing
	bool emptyEpoch = true;
#	ifdef ENABLE_PARALLELISATION
#	ifndef ENABLE_UNIT_TESTS
		Eigen::setNbThreads(1);
#		pragma omp parallel for
#	endif
#	endif
	for (int i = 0; i < stationMap.size(); i++)
	{
		auto rec_ptr_iterator = stationMap.begin();
		std::advance(rec_ptr_iterator, i);

		auto& [id, rec] = *rec_ptr_iterator;
		mainOncePerEpochPerStation(rec, net, emptyEpoch);
	}
	Eigen::setNbThreads(0);

	if	(emptyEpoch)
	{
		BOOST_LOG_TRIVIAL(warning)
		<< "Warning: Epoch " << epoch << " has no observations";
	}

	if (acsConfig.process_ppp)
	{
		PPP(netTrace, stationMap, net.kfState);
	}

	mongoMeasSatStat(stationMap);

	if (acsConfig.output_rinex_nav)
	{
		writeRinexNav(acsConfig.rinex_nav_version);
	}

	if (acsConfig.process_network)
	{
		networkEstimator(netTrace, stationMap, net.kfState, tsync);
	}

	if (1)	//acsConfig.ambiguityResolution
	{
		//ambiguityResolution(netTrace, net.kfState);
	}
	
	if (acsConfig.delete_old_ephemerides)
	{
		cullOldEphs(tsync);
		cullOldSSRs(tsync);
	}

	if (acsConfig.check_plumbing)
	{
		plumber();
	}
	
	mainPerEpochPostProcessingAndOutputs(net, stationMap);
	
	if (acsConfig.process_ionosphere)
	{
		updateIonosphereModel(netTrace, net.kfState.metaDataMap[IONSTEC_FILENAME_STR], net.kfState.metaDataMap[IONEX_FILENAME_STR], stationMap, time);
	}
	
	TestStack::testStatus();

	callbacksOncePerEpoch();
}

void mainPostProcessing(
	Network&	net,
	StationMap&	stationMap)
{
	BOOST_LOG_TRIVIAL(info)
	<< "Post processing... ";
	
	auto netTrace = getTraceFile(net);
	
	if	( ( acsConfig.process_network
		  ||acsConfig.process_ppp)
		&&	acsConfig.process_minimum_constraints
		&&	acsConfig.minCOpts.once_per_epoch == false)
	{
		BOOST_LOG_TRIVIAL(info)
		<< std::endl
		<< "---------------PROCESSING NETWORK WITH MINIMUM CONSTRAINTS ------------- " << std::endl;
		
		for (auto& [id, rec] : stationMap)
		{
			rec.minconApriori = rec.aprioriPos;
		}
			
		mincon(netTrace, net.kfState);
	}
	
	if	(  acsConfig.ambrOpts.mode	!= +E_ARmode::OFF 
		&& acsConfig.ionoOpts.corr_mode	== +E_IonoMode::IONO_FREE_LINEAR_COMBO)
	{
		dump_WLambg(netTrace);
	}
	
	if (acsConfig.output_sinex)
	{
		sinexPostProcessing(tsync, stationMap, net.kfState);
	}
	
	if (acsConfig.output_persistance)
	{
		BOOST_LOG_TRIVIAL(info)
		<< "Storing persistant states to continue processing...";

		outputPersistanceStates(stationMap, net.kfState);
	}

	if	( acsConfig.process_network
		||acsConfig.process_ppp)
	{
// 		outputMqtt	(net.kfState);
		outputOrbit	(net.kfState);
	}
	
	outputPredictedStates(netTrace, net, acsConfig.localMongo);
	
	if (acsConfig.output_predicted_orbits)
	{
		outputMongoOrbits();
	}
	
	if (acsConfig.process_rts)
	{
		if	(  acsConfig.process_user
			&& acsConfig.pppOpts.rts_lag < 0)
		for (auto& [id, rec] : stationMap)
		{
			BOOST_LOG_TRIVIAL(info)
			<< std::endl
			<< "---------------PROCESSING PPP WITH RTS------------------------- " << std::endl;

			RTS_Process(rec.pppState,	true, &stationMap);
		}
		
		if	( acsConfig.process_network
			&&acsConfig.pppOpts.rts_lag < 0)
		{
			BOOST_LOG_TRIVIAL(info)
			<< std::endl
			<< "---------------PROCESSING NETWORK WITH RTS--------------------- " << std::endl;

			RTS_Process(net.kfState,		true, &stationMap);
		}
		
		if	( acsConfig.process_ppp
			&&acsConfig.pppOpts.rts_lag < 0)
		{
			BOOST_LOG_TRIVIAL(info)
			<< std::endl
			<< "---------------PROCESSING PPPPP WITH RTS--------------------- " << std::endl;

			RTS_Process(net.kfState,		true, &stationMap);
		}

		if	( acsConfig.process_ionosphere
			&&acsConfig.pppOpts.rts_lag < 0)
		{
			BOOST_LOG_TRIVIAL(info)
			<< std::endl
			<< "---------------PROCESSING IONOSPHERE WITH RTS------------------ " << std::endl;

			RTS_Process(iono_KFState,		true, &stationMap);
		}
	}

	if (acsConfig.testOpts.enable)
	{
		TestStack::printStatus(true);
		TestStack::saveData();
	}

	Instrument::printStatus();

	outputSummaries(netTrace, stationMap);
	
	outputStatistics(netTrace, net.kfState.statisticsMapSum, net.kfState.statisticsMapSum);
}

int ginan(
	int		argc, 
	char**	argv)
{
	tracelevel(5);
	
	// Register the sink in the logging core
	boost::log::core::get()->add_sink(boost::make_shared<sinks::synchronous_sink<ConsoleLog>>());
	boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
	
	BOOST_LOG_TRIVIAL(info)
	<< "PEA starting... (" << ginanBranchName() << " " << ginanCommitVersion() << " from " << ginanCommitDate() << ")" << std::endl;

	GTime peaStartTime = timeGet();

	exitOnErrors();
	
	bool pass = configure(argc, argv);
	if (pass == false)
	{
		BOOST_LOG_TRIVIAL(error) 	<< "Error: Incorrect configuration";
		BOOST_LOG_TRIVIAL(info) 	<< "PEA finished";
		NtripSocket::io_service.stop();
		return EXIT_FAILURE;
	}
	
	if (acsConfig.output_log)
	{
		addFileLog();
	}
	

	TestStack::openData();

	BOOST_LOG_TRIVIAL(info)
	<< "Threading with " << Eigen::nbThreads()
	<< " threads" << std::endl;

	

	BOOST_LOG_TRIVIAL(info)
	<< "Logging with trace level:" << acsConfig.trace_level << std::endl << std::endl;

	tracelevel(acsConfig.trace_level);

	if (acsConfig.input_persistance)
	{
		BOOST_LOG_TRIVIAL(info)
		<< "Loading persistant navigation object to continue processing...";

		inputPersistanceNav();
	}

	if (acsConfig.process_ionosphere)
	{
		bool pass = configIonModel();
		if	(pass == false)
		{
			BOOST_LOG_TRIVIAL(error)
			<< "Error in Ionosphere Model configuration";

			return EXIT_FAILURE;
		}
	}
	
	if (acsConfig.ambrOpts.mode != +E_ARmode::OFF)
	{
		config_AmbigResl();
	}
	
	//prepare the satNavMap so that it at least has entries for everything
	for (auto [sys, max] :	{	tuple<E_Sys, int>{E_Sys::GPS, NSATGPS},
								tuple<E_Sys, int>{E_Sys::GLO, NSATGLO},
								tuple<E_Sys, int>{E_Sys::GAL, NSATGAL},
								tuple<E_Sys, int>{E_Sys::QZS, NSATQZS},
								tuple<E_Sys, int>{E_Sys::LEO, NSATLEO},
								tuple<E_Sys, int>{E_Sys::BDS, NSATBDS},
								tuple<E_Sys, int>{E_Sys::SBS, NSATSBS}})
	for (int prn = 1; prn <= max; prn++)
	{
		SatSys Sat(sys, prn);
		
		if (acsConfig.process_sys[Sat.sys] == false)
			continue;
	
		auto& satOpts = acsConfig.getSatOpts(Sat);
		
		if (satOpts.exclude)
			continue;
		
		nav.satNavMap[Sat].id = Sat.id();
	}

	Network net;
	{
		net.kfState.id						= "Net";
		net.kfState.max_filter_iter			= acsConfig.pppOpts.max_filter_iter;
		net.kfState.max_prefit_remv			= acsConfig.pppOpts.max_prefit_remv;
		net.kfState.inverter				= acsConfig.pppOpts.inverter;
		net.kfState.sigma_check				= acsConfig.pppOpts.sigma_check;
		net.kfState.sigma_threshold			= acsConfig.pppOpts.sigma_threshold;
		net.kfState.w_test					= acsConfig.pppOpts.w_test;
		net.kfState.chi_square_test			= acsConfig.pppOpts.chi_square_test;
		net.kfState.chi_square_mode			= acsConfig.pppOpts.chi_square_mode;
		net.kfState.simulate_filter_only	= acsConfig.pppOpts.simulate_filter_only;
		net.kfState.output_residuals		= acsConfig.output_residuals;
		net.kfState.outputMongoMeasurements	= acsConfig.localMongo.output_measurements;
		net.kfState.measRejectCallbacks	.push_back(deweightMeas);
		net.kfState.measRejectCallbacks	.push_back(incrementPhaseSignalError);
		net.kfState.measRejectCallbacks	.push_back(pseudoMeasTest);
		
		net.kfState.stateRejectCallbacks.push_back(rejectByState);
		net.kfState.stateRejectCallbacks.push_back(orbitGlitchReaction);
	}

	if	(  acsConfig.process_rts
		&& acsConfig.pppOpts.rts_lag)
	{
		net.kfState.rts_lag		= acsConfig.pppOpts.rts_lag;
		iono_KFState.rts_lag	= acsConfig.pppOpts.rts_lag;
	}

	if	(acsConfig.process_ionosphere)
	{
		iono_KFState.measRejectCallbacks	.push_back(deweightMeas);
		iono_KFState.stateRejectCallbacks	.push_back(rejectByState);
	}
	//initialise mongo
	mongoooo();
	
	for (auto once : {1})
	{
		if (acsConfig.yamls.empty())
		{
			continue;
		}
		
		YAML::Emitter emitter;
		emitter << YAML::DoubleQuoted << YAML::Flow << YAML::BeginSeq << acsConfig.yamls[0];
	
		string config(emitter.c_str() + 1);
		
		mongoOutputConfig(config);
	}
		
		
	if (acsConfig.rts_only)
	{
		net.kfState.rts_lag = 4000;
		net.kfState.rts_basename = acsConfig.pppOpts.rts_filename;
		
		RTS_Process(net.kfState);
		
		exit(0);
	}
	
	if (acsConfig.input_persistance)
	{
		BOOST_LOG_TRIVIAL(info)
		<< "Loading persistant states to continue processing...";

		inputPersistanceStates(stationMap, net.kfState);
	}
	
	initialiseBiasSinex();

	boost::posix_time::ptime logptime = currentLogptime();
	createDirectories(logptime);
	
	reloadInputFiles();

	addDefaultBiasSinex();

	NtripSocket::startClients();

	configureUploadingStreams();

	if (acsConfig.start_epoch.is_not_a_date_time() == false)
	{
		PTime startTime;
		startTime.bigTime = boost::posix_time::to_time_t(acsConfig.start_epoch);
		
		tsync = startTime;
	}
	
	createTracefiles(stationMap, net);
	
	if (acsConfig.mincon_only)
	{
		minconOnly(std::cout, stationMap);
	}
	
	doDebugs();

	BOOST_LOG_TRIVIAL(info)
	<< std::endl;
	BOOST_LOG_TRIVIAL(info)
	<< "Starting to process epochs...";
	
	//============================================================================
	// MAIN PROCESSING LOOP														//
	//============================================================================

	// Read the observations for each station and do stuff
	bool	complete					= false;							// When all input files are empty the processing is deemed complete - run until then, or until something else breaks the loop
	int		loopEpochs					= 0;								// A count of how many loops of epoch_interval this loop used up (usually one, but may be more if skipping epochs)
	auto	nextNominalLoopStartTime	= system_clock::now() + 10s;		// The time the next loop is expected to start - if it doesnt start until after this, it may be skipped
	while (complete == false)
	{
		if (tsync != GTime::noTime())
		{
			tsync.bigTime			+= loopEpochs * acsConfig.epoch_interval;
			
			if (fabs(tsync.bigTime - round(tsync.bigTime)) < acsConfig.epoch_tolerance)
			{
				tsync.bigTime = round(tsync.bigTime);
			}
		}

		epoch						+= loopEpochs;
		nextNominalLoopStartTime	+= loopEpochs * std::chrono::milliseconds((int)(acsConfig.wait_next_epoch * 1000));

		// Calculate the time at which we will stop waiting for data to come in for this epoch
		auto breakTime	= nextNominalLoopStartTime
						+ std::chrono::milliseconds((int)(acsConfig.wait_all_stations	* 1000));

		BOOST_LOG_TRIVIAL(info) << std::endl
		<< "Starting epoch #" << epoch;

		for (auto& [id, rec] : stationMap)
		{
			rec.ready = false;
			
			auto trace		= getTraceFile(rec);

			trace		<< std::endl << "------=============== Epoch " << epoch	<< " =============-----------" << std::endl;
			trace		<< std::endl << "------=============== Time  " << tsync	<< " =============-----------" << std::endl;
		}
		
		{
			auto netTrace	= getTraceFile(net);
			
			netTrace	<< std::endl << "------=============== Epoch " << epoch	<< " =============-----------" << std::endl;
			netTrace	<< std::endl << "------=============== Time  " << tsync	<< " =============-----------" << std::endl;
		}

		//get observations from streams (allow some delay between stations, and retry, to ensure all messages for the epoch have arrived)
		map<string, bool>	dataAvailableMap;
		bool 				foundFirst	= false;
		bool				repeat		= true;
		bool				atLeastOnce	= true;
		while	(   atLeastOnce
				||( repeat
				  &&system_clock::now() < breakTime))
		{
			atLeastOnce = false;
			
			if (acsConfig.require_obs)
			{
				repeat = false;
			}

			//remove any dead streams
			for (auto iter = streamParserMultimap.begin(); iter != streamParserMultimap.end(); )
			{
				auto& [id, streamParser_ptr]	= *iter;
				auto& stream					= streamParser_ptr->stream;
				
				try
				{
					auto& obsStream	= dynamic_cast<ObsStream&>(*streamParser_ptr);
					
					if (obsStream.hasObs())
					{
						iter++;
						continue;
					}
				}
				catch(...){}
				
				if (stream.isDead())
				{
					BOOST_LOG_TRIVIAL(info)
					<< "No more data available on " << stream.sourceString << std::endl;

					//record as dead and erase
					streamDOAMap[stream.sourceString] = true;

					iter = streamParserMultimap.erase(iter);
					stationMap[id].obsList.clear();
				}
				else
				{
					iter++;
				}
			}

			if (streamParserMultimap.empty())
			{
				static bool once = true;
				if (once)
				{
					once = false;
					
					BOOST_LOG_TRIVIAL(info)
					<< std::endl;
					BOOST_LOG_TRIVIAL(info)
					<< "Inputs finished at epoch #" << epoch;
				}

				if (acsConfig.require_obs)
					complete = true;

				break;
			}

			//parse all non-observation streams
			for (auto& [id, streamParser_ptr] : streamParserMultimap)
			try
			{
				auto& obsStream = dynamic_cast<ObsStream&>(*streamParser_ptr);
			}
			catch (std::bad_cast& e)
			{
				streamParser_ptr->parse();
			}
			
			for (auto& [id, streamParser_ptr] : streamParserMultimap)
			{
				ObsStream* obsStream_ptr;
				
				try
				{
					obsStream_ptr = &dynamic_cast<ObsStream&>(*streamParser_ptr);
				}
				catch (std::bad_cast& e)
				{
					continue;
				}
				
				auto& obsStream = *obsStream_ptr;
				
				auto& recOpts = acsConfig.getRecOpts(id);

				if (recOpts.exclude)
				{
					continue;
				}

				auto& rec = stationMap[id];

				//try to get some data (again)
				if (rec.ready == false)
				{
					bool moreData = true;
					while (moreData)
					{
						if (acsConfig.assign_closest_epoch)	rec.obsList = obsStream.getObs(tsync, acsConfig.epoch_interval / 2);
						else								rec.obsList = obsStream.getObs(tsync, acsConfig.epoch_tolerance);

						switch (obsStream.obsWaitCode)
						{
							case E_ObsWaitCode::EARLY_DATA:								preprocessor(net, rec);	break;
							case E_ObsWaitCode::OK:					moreData = false;	preprocessor(net, rec);	break;
							case E_ObsWaitCode::NO_DATA_WAIT:		moreData = false;							break;
							case E_ObsWaitCode::NO_DATA_EVER:		moreData = false;							break;
						}
					}
				}
				
				if (rec.obsList.empty())
				{
					//failed to get observations
					if (obsStream.obsWaitCode == +E_ObsWaitCode::NO_DATA_WAIT)
					{
						// try again later
						repeat = true;
						sleep_for(1ms);
					}
					
					continue;
				}
				
				if (tsync == GTime::noTime())
				{
					tsync = rec.obsList.front()->time.floorTime(acsConfig.epoch_interval);
					
					acsConfig.start_epoch	= boost::posix_time::from_time_t((time_t)((PTime)tsync).bigTime);
					
					if (tsync + acsConfig.epoch_tolerance < rec.obsList.front()->time)
					{
						repeat = true;
						continue;	
					}
				}

				dataAvailableMap[rec.id] = true;
				
				if (foundFirst == false)
				{
					foundFirst = true;

					//first observation found for this epoch, give any other stations some time to get their observations too
					//only shorten waiting periods, never extend
					auto now = system_clock::now();

					auto alternateBreakTime = now + std::chrono::milliseconds((int)(acsConfig.wait_all_stations	* 1000));
					auto alternateStartTime = now + std::chrono::milliseconds((int)(acsConfig.wait_next_epoch	* 1000));

					if (alternateBreakTime < breakTime)						{	breakTime					= alternateBreakTime;	}
					if (alternateStartTime < nextNominalLoopStartTime)		{	nextNominalLoopStartTime	= alternateStartTime;	}
				}
				
				rec.ready = true;
			}
		}

		if (complete)
		{
			break;
		}

		if (tsync == GTime::noTime())
		{
			if (acsConfig.require_obs)
				continue;
			
			tsync = timeGet();
		}
		
		BOOST_LOG_TRIVIAL(info)
		<< "Synced " << dataAvailableMap.size() << " stations...";
		
		GTime epochStartTime	= timeGet();
		{
			mainOncePerEpoch(net, stationMap, tsync);
		}
		GTime epochStopTime		= timeGet();

		
		
		GWeek	week	= tsync;
		GTow	tow		= tsync;
		
		int fractionalMilliseconds = (tsync.bigTime - (long int) tsync.bigTime) * 1000;
		auto boostTime = boost::posix_time::from_time_t((time_t)((PTime)tsync).bigTime) + boost::posix_time::millisec(fractionalMilliseconds);
	
		BOOST_LOG_TRIVIAL(info)
		<< "Processed epoch #" << epoch
		<< " - " << "GPS time: " << week << " " << std::setw(6) << tow << " - " << boostTime 
		<< " (took " << (epochStopTime-epochStartTime) << ")";

		// Check end epoch
		if	(  acsConfig.end_epoch.is_not_a_date_time() == false
			&& boostTime >= acsConfig.end_epoch)
		{
			BOOST_LOG_TRIVIAL(info)
			<< "Exiting at epoch " << epoch << " (" << boostTime
			<< ") as end epoch " << acsConfig.end_epoch
			<< " has been reached";

			break;
		}

		// Check number of epochs
		if	(  acsConfig.max_epochs	> 0
			&& epoch					>= acsConfig.max_epochs)
		{
			BOOST_LOG_TRIVIAL(info)
			<< std::endl
			<< "Exiting at epoch " << epoch << " (" << boostTime
			<< ") as epoch count " << acsConfig.max_epochs
			<< " has been reached";

			break;
		}

		// Calculate how many loops need to be skipped based on when the next loop was supposed to begin
		auto loopStopTime		= system_clock::now();
		auto loopExcessDuration = loopStopTime - (nextNominalLoopStartTime + std::chrono::milliseconds((int)(acsConfig.wait_all_stations * 1000)));
		int excessLoops			= loopExcessDuration / std::chrono::milliseconds((int)(acsConfig.wait_next_epoch * 1000));

		if (excessLoops < 0)		{	excessLoops = 0;	}
		if (excessLoops > 0)	
		{
			BOOST_LOG_TRIVIAL(warning) << std::endl 
			<< "Warning: Excessive time elapsed, skipping " << excessLoops 
			<< " epochs to epoch " << epoch + excessLoops + 1 
			<< ". Configuration 'wait_next_epoch' is " << acsConfig.wait_next_epoch;
		}

		loopEpochs = 1 + excessLoops;
	}

	// Disconnect the downloading clients and stop the io_service for clean shutdown.
	for (auto& [id, ntripStream] : only<NtripStream>(streamParserMultimap))
	{
		ntripStream.disconnect();
	}
	
	ntripBroadcaster.stopBroadcast();
	NtripSocket::io_service.stop();

	GTime peaInterTime = timeGet();
	BOOST_LOG_TRIVIAL(info)
	<< std::endl
	<< "PEA started  processing at : " << peaStartTime << std::endl
	<< "and finished processing at : " << peaInterTime << std::endl
	<< "Total processing duration  : " << (peaInterTime - peaStartTime) << std::endl << std::endl;

	BOOST_LOG_TRIVIAL(info)
	<< std::endl
	<< "Finalising streams and post processing...";

	mainPostProcessing(net, stationMap);

	GTime peaStopTime = timeGet();
	BOOST_LOG_TRIVIAL(info)
	<< std::endl
	<< "PEA started  processing at : " << peaStartTime	<< std::endl
	<< "and finished processing at : " << peaStopTime	<< std::endl
	<< "Total processing duration  : " << (peaStopTime - peaStartTime) << std::endl << std::endl;

	BOOST_LOG_TRIVIAL(info)
	<< "PEA finished";

	return EXIT_SUCCESS;
}

