
// #pragma GCC optimize ("O0")

#include <boost/log/trivial.hpp>

#include "acsConfig.hpp"
#include "satStat.hpp"
#include "station.hpp"
#include "algebra.hpp"
#include "ppp.hpp"

/** Deweight worst measurement
 */
bool deweightMeas(
	Trace&		trace,
	KFState&	kfState,
	KFMeas&		kfMeas,
	int			index)
{
	if (acsConfig.deweight_factor == 0)
	{
		return true;
	}
	
	trace << std::endl << "Deweighting " << kfMeas.obsKeys[index] << std::endl;

	kfState.statisticsMap["Meas deweight"]++;

	kfMeas.R.row(index) *= acsConfig.deweight_factor;
	kfMeas.R.col(index) *= acsConfig.deweight_factor;
	
	return true;
}

/** Call state rejection functions when a measurement is a pseudo observation
 */
bool pseudoMeasTest(
	Trace&		trace,
	KFState&	kfState,
	KFMeas&		kfMeas,
	int			index)
{
	if (kfMeas.metaDataMaps[index]["pseudoObs"] == (void*) false)
	{
		return true;
	}
	
	for (auto& [key, state] : kfState.kfIndexMap)
	{
		if	( kfMeas.H(index, state)
			&&key.type == KF::ORBIT)
		{
			orbitGlitchReaction(trace, kfState, kfMeas, key);
		}
	}
	
	return true;
}

/** Deweight measurement and its relatives
 */
bool deweightStationMeas(
	Trace&		trace,
	KFState&	kfState,
	KFMeas&		kfMeas,
	int			index)
{
	string id = kfMeas.obsKeys[index].str;
	
	for (int i = 0; i < kfMeas.obsKeys.size(); i++)
	{
		auto& obsKey = kfMeas.obsKeys[i];
		
		if (obsKey.str != id)
		{
			continue;
		}
		
		trace << std::endl << "Deweighting " << kfMeas.obsKeys[i] << std::endl;

		kfState.statisticsMap["Station deweight"]++;

		kfMeas.R.row(i) *= acsConfig.deweight_factor;
		kfMeas.R.col(i) *= acsConfig.deweight_factor;
		
		map<string, void*>& metaDataMap = kfMeas.metaDataMaps[i];

		bool* used_ptr = (bool*) metaDataMap["used_ptr"];
		
		if (used_ptr)
		{
			*used_ptr = false;	
		}
	}
	return true;
}

/** Count worst measurement
 */
bool incrementPhaseSignalError(
	Trace&		trace,
	KFState&	kfState,
	KFMeas&		kfMeas,
	int			index)
{
	map<string, void*>& metaDataMap = kfMeas.metaDataMaps[index];

	unsigned int* PhaseRejectCount_ptr = (unsigned int*) metaDataMap["phaseRejectCount"];

	if (PhaseRejectCount_ptr == nullptr)
	{
		return true;
	}

	unsigned int&	phaseRejectCount	= *PhaseRejectCount_ptr;

	//increment counter, and clear the pointer so it cant be reset to zero in subsequent operations (because this is a failure)
	phaseRejectCount++;
	metaDataMap["phaseRejectCount"] = nullptr;
	
	trace << std::endl << "Incrementing phaseRejectCount on " << kfMeas.obsKeys[index].Sat.id() << " to " << phaseRejectCount;
	
	return true;
}

bool countSignalErrors(
	Trace&		trace,
	KFState&	kfState,
	KFMeas&		kfMeas,
	int			index)
{
	map<string, void*>& metaDataMap = kfMeas.metaDataMaps[index];

	GObs* obs_ptr = (GObs*) metaDataMap["obs_ptr"];

	if (obs_ptr == nullptr)
	{
		return true;
	}

	KFKey&		obsKey	= kfMeas.obsKeys[index];
	GObs&		obs		= *obs_ptr;

	if (obsKey.type == KF::PHAS_MEAS)
	{
		//this is a phase observation
		obs.Sigs[(E_FType)obsKey.num].phaseError = true;
	}

	return true;
}


bool resetPhaseSignalError(
	KFMeas&		kfMeas,
	int			index)
{
	map<string, void*>& metaDataMap = kfMeas.metaDataMaps[index];

	//these will have been set to null if there was an error after adding the measurement to the list
	for (auto suffix : {"", "_alt"})
	{
		unsigned int* phaseRejectCount_ptr = (unsigned int*) metaDataMap[(string)"phaseRejectCount" + suffix];

		if (phaseRejectCount_ptr == nullptr)
		{
			return true;
		}

		unsigned int&	phaseRejectCount	= *phaseRejectCount_ptr;

		phaseRejectCount = 0;
	}
	
	return true;
}


bool resetPhaseSignalOutage(
	KFMeas&		kfMeas,
	int			index)
{
	map<string, void*>& metaDataMap = kfMeas.metaDataMaps[index];

	for (auto suffix : {"", "_alt"})
	{
		unsigned int* phaseOutageCount_ptr = (unsigned int*) metaDataMap[(string)"phaseOutageCount" + suffix];

		if (phaseOutageCount_ptr == nullptr)
		{
			return true;
		}

		unsigned int&	phaseOutageCount	= *phaseOutageCount_ptr;
		
		phaseOutageCount = 0;
	}
	
	return true;
}

bool resetIonoSignalOutage(
	KFMeas&		kfMeas,
	int			index)
{
	map<string, void*>& metaDataMap = kfMeas.metaDataMaps[index];

	for (auto suffix : {"", "_alt"})
	{
		unsigned int* ionoOutageCount_ptr = (unsigned int*) metaDataMap[(string)"ionoOutageCount" + suffix];

		if (ionoOutageCount_ptr == nullptr)
		{
			return true;
		}

		unsigned int&	ionoOutageCount	= *ionoOutageCount_ptr;
		
		ionoOutageCount = 0;
	}
	
	return true;
}

/** Reject measurements attached to worst state using measurement reject callback list
 */
bool rejectByState(
			Trace&		trace,
			KFState&	kfState,
			KFMeas&		kfMeas,
	const	KFKey&		kfKey)
{
	if (acsConfig.reject_on_state_error == false)
	{
		return true;
	}
	
	trace << std::endl << "Bad state detected " << kfKey << " - rejecting all referencing measurements" << std::endl;

	kfState.statisticsMap["State rejection"]++;
	
	int stateIndex = kfState.getKFIndex(kfKey);
	
	for (int meas = 0; meas < kfMeas.H.rows(); meas++)
	{
		if (kfMeas.H(meas, stateIndex))
		{
			kfState.doMeasRejectCallbacks(trace, kfMeas, meas);
		}
	}
	
	return true;
}

/** Remove any states connected to a bad clock if it glitches
 */
bool clockGlitchReaction(
			Trace&		trace,
			KFState&	kfState,
			KFMeas&		kfMeas,
	const	KFKey&		kfKey)
{
	if	(  kfKey.type != KF::SAT_CLOCK
		&& kfKey.type != KF::REC_SYS_BIAS)
	{
		return true;
	}
	
	if (acsConfig.reinit_on_clock_error == false)
	{
		return true;
	}
	
	trace << std::endl << "Bad clock detected " << kfKey << " - resetting linked states" << std::endl;

	kfState.statisticsMap["Clock glitch"]++;
	
	for (auto& [key, index] : kfState.kfIndexMap)
	{
		if	(  kfKey.type	== KF::SAT_CLOCK
			&& kfKey.Sat	== key.Sat
			&&( key	.type	== KF::AMBIGUITY
			  ||key	.type	== KF::SAT_CLOCK))
		{
			//remove the satellite clock, and any ambiguities that are connected to it.
			trace << "- Removing " << key << std::endl;
			
			kfState.removeState(key);
		}
		
		if	(  kfKey.type	== KF::REC_SYS_BIAS
			&& kfKey.str	== key.str
			&&( key	.type	== KF::AMBIGUITY
			  ||key	.type	== KF::REC_SYS_BIAS))
		{
			//remove the satellite clock, and any ambiguities that are connected to it.
			trace << "- Removing " << key << std::endl;
			
			kfState.removeState(key);
			
			if (kfKey.rec_ptr)
			{
				//make sure receiver clock corrections get reset too.
				trace << "- Resetting clock adjustment" << std::endl;
				
				auto& rec = *kfKey.rec_ptr;
				
				rec.sol.deltaDt_net_old[E_Sys::GPS] = 0;
			}
		}
	}
	
	return true;
}


bool orbitGlitchReaction(
			Trace&		trace,
			KFState&	kfState,
			KFMeas&		kfMeas,
	const	KFKey&		kfKey)
{
	if (kfKey.type != KF::ORBIT)
	{
		return true;
	}
	
	trace << std::endl << "Bad orbit state detected " << kfKey;

	kfState.statisticsMap["Orbit state reject"]++;
	
	Exponential exponentialNoise;
	exponentialNoise.tau	=		acsConfig.orbit_vel_proc_noise_trail_tau;
	exponentialNoise.value	= SQR(	acsConfig.orbit_vel_proc_noise_trail);
	
	MatrixXd F = MatrixXd::Identity	(kfState.x.rows(), kfState.x.rows());
	MatrixXd Q = MatrixXd::Zero		(kfState.x.rows(), kfState.x.rows());
	
	for (auto& [key, index] : kfState.kfIndexMap)
	{
		if	(  key.type	== KF::ORBIT
			&& key.str	== kfKey.str
			&& key.Sat	== kfKey.Sat
			&& key.num	<  3)
		{
			Q(index, index) = SQR(acsConfig.orbit_pos_proc_noise);
		}
		
		if	(  key.type	== KF::SAT_POS_RATE
			&& key.str	== kfKey.str
			&& key.Sat	== kfKey.Sat
			&& key.num	>= 3)
		{
			Q(index, index) = SQR(acsConfig.orbit_vel_proc_noise);
	
			kfState.setExponentialNoise(key, exponentialNoise);
		}
	}
	
	kfState.manualStateTransition(trace, kfState.time, F, Q);
	
	return false;
}

