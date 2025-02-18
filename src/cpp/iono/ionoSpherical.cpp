
#include "acceleration.hpp"
#include "observations.hpp"
#include "coordinates.hpp"
#include "corrections.hpp"
#include "acsConfig.hpp"
#include "constants.hpp"
#include "ionoModel.hpp"
#include "planets.hpp"
#include "common.hpp"


Legendre ionLeg;                              		
double ionSPHLastColatitude = -100;

struct SphBasis
{
	int layer	= 0;					
	int degree	= 0;					
	int order	= 0;					
	E_TrigType trigType;
};

map<int, SphBasis>									sphBasisMap;
map<int, map<int, map<int, map<E_TrigType, int>>>>	sphBasisIndexMaps;


/** configures the spherical harmonics model.
Specifically it initializes:
shar_valid				time validity of a rotation matrix (the rotation matrix will chase the sun position)
Sph_Basis_list			List of ionosphere basis
time:  		I 		time of observations (to update the rotation matrix)
IPP: 			I 		Ionospheric piercing point to be updated
Since the spherical harmonic model has global validity, the check always return 1
-----------------------------------------------------
Author: Ken Harima @ RMIT 29 July 2020
-----------------------------------------------------*/
int configIonModelSphhar()
{
	if		(acsConfig.ionModelOpts.function_degree == 0)										acsConfig.ionModelOpts.function_degree	= acsConfig.ionModelOpts.function_order;	
	else if (acsConfig.ionModelOpts.function_order > acsConfig.ionModelOpts.function_degree)	acsConfig.ionModelOpts.function_order	= acsConfig.ionModelOpts.function_degree;
	
	int Nmax = acsConfig.ionModelOpts.function_degree + 1;
	int nlay = acsConfig.ionModelOpts.layer_heights.size();
	if (nlay == 0) 
	{
		acsConfig.ionModelOpts.layer_heights.push_back(350);
		nlay = 1;
	}
	
	ionLeg.setNmax(Nmax);

	int ind = 0;
	for (int layer	= 0;		layer	< nlay;										layer++)
	for (int order	= 0;		order	< acsConfig.ionModelOpts.function_order;	order++)
	for (int degree	= order;	degree	< acsConfig.ionModelOpts.function_degree;	degree++)
	{
		SphBasis basis;
		basis.layer		= layer;
		basis.order		= order;
		basis.degree	= degree;
			
// 		if (1)
		{
			basis.trigType	= E_TrigType::COS;
			
			sphBasisIndexMaps[layer][degree][order][basis.trigType]	= ind;
			sphBasisMap[ind]										= basis;
		
			ind++;
		}
		
		if (order > 0)
		{
			basis.trigType	= E_TrigType::SIN;
			
			sphBasisIndexMaps[layer][degree][order][basis.trigType]	= ind;
			sphBasisMap[ind]										= basis;
			
			ind++;
		}
	}

	acsConfig.ionModelOpts.numBasis = ind;

	return ind;
}

/** rotates the Ionosphere piercing point
time:  		I 		time of observations (to update the rotation matrix)
IPP: 			I 		Ionospheric piercing point to be updated
Since the spherical harmonic model has global validity, the check always return 1
-----------------------------------------------------
Author: Ken Harima @ RMIT 29 July 2020
-----------------------------------------------------*/
bool ippCheckSphhar(
	GTime		time, 
	VectorPos&	ionPP)
{
	if 	(time == GTime::noTime()) 
		return false;

	if (acsConfig.ionModelOpts.use_rotation_mtx)
	{
		static Matrix3d	sphRotMatrix;				/* Rotation matrix (to centre of map) */
		static GTime	sphTime;
		
		double	sphValid	= 10;
		
		if	(  sphTime == GTime::noTime()
			|| fabs((time - sphTime).to_double()) > sphValid )
		{
			VectorEcef	rSun;
			planetPosEcef(time, E_ThirdBody::SUN, rSun);
		
			VectorPos sunpos = ecef2pos(rSun);

			double lat = sunpos.lat();
			double lon = sunpos.lon();
			
			sphRotMatrix	= Eigen::AngleAxisd(lon, Vector3d::UnitZ()) * Eigen::AngleAxisd(-lat, Vector3d::UnitY());
		
			sphTime			= time;
		}

		VectorPos pos;
		pos.hgt() = ionPP.hgt();
		pos.lon() = ionPP.lon();
		pos.hgt() = acsConfig.ionModelOpts.layer_heights[0];
		
		VectorEcef rpp = pos2ecef(pos);
		
		VectorEcef rrot = (Vector3d)(sphRotMatrix * rpp);
		pos = ecef2pos(rrot);
	
		ionPP.lat() = pos.lat() + PI/2;			/* colatitude for spherical harmonics */
		ionPP.lon() = pos.lon();
	}
	else
	{
		double tow = GTow(time);
		
		ionPP.lat() = ionPP.lat() - PI/2;
		ionPP.lon()+= (tow-50400)*PI/43200;
		double day= floor(ionPP.lon()/(2*PI))*2*PI;
																		
		ionPP.lon()-= day;
	}
	return true;
}

/** Evaluates spherical harmonics basis functions
	int ind			I		
	obs				I		Ionosphere measurement struct
		latIPP				- Latitude of Ionosphere Piercing Point
		lonIPP				- Longitude of Ionosphere Piercing Point
		angIPP				- Angular gain for Ionosphere Piercing Point
	int slant		I		0: coefficient for VTEC, 1: coefficient for STEC
----------------------------------------------------------------------------*/
double ionCoefSphhar(
	int			ind,			///< Basis function number
	IonoObs&	obs,			///< Ionospheric observation metadata	
	bool		slant)			///< apply slant factor, false: coefficient for VTEC, true: coefficient for STEC
{
	if (ind >= sphBasisMap.size())
		return 0;

	auto& basis = sphBasisMap[ind];

	if (basis.order		> acsConfig.ionModelOpts.function_order)
		return 0;

	if (basis.degree	> acsConfig.ionModelOpts.function_degree)
		return 0;

	double colat = obs.ippMap[basis.layer].lat;
	
	if (fabs(ionSPHLastColatitude - colat) > 0.01)
		ionLeg.calculate(cos(colat));

	double coeff = pow(-1, basis.order) * ionLeg.Pnm(basis.degree, basis.order);
	
	double angle = basis.order * obs.ippMap[basis.layer].lon;
	
	if		(basis.trigType == +E_TrigType::SIN)		coeff *= sin(angle);
	else if (basis.trigType == +E_TrigType::COS)		coeff *= cos(angle);

	if (slant)
	{
		coeff *= obs.ippMap[basis.layer].slantFactor;
	}

	return coeff;
}

/** Estimate Ionosphere VTEC using Spherical Cap Harmonic models
	gtime_t  time		I		time of solutions (not useful for this one
	Ion_pp				I		Ionosphere Piercing Point
	layer				I 		Layer number
	vari				O		variance of VTEC
returns: VETC at piercing point
----------------------------------------------------------------------------*/
double ionVtecSphhar(
	GTime		time,
	VectorPos&	ionPP,
	int			layer,
	double&		var,
	KFState&	kfState)
{
	VectorPos ionpp_cpy;

	ionpp_cpy[0] = ionPP[0];
	ionpp_cpy[1] = ionPP[1];
	ionpp_cpy[2] = acsConfig.ionModelOpts.layer_heights[layer];

	ippCheckSphhar(time, ionpp_cpy);

	var = 0;
	
	IonoObs tmpobs;
	tmpobs.ippMap[layer].lat			= ionpp_cpy[0];
	tmpobs.ippMap[layer].lon			= ionpp_cpy[1];
	tmpobs.ippMap[layer].slantFactor	= 1;

	double iono = 0;
	
	for (int basisNum = 0; basisNum < acsConfig.ionModelOpts.numBasis; basisNum++)
	{
		auto& basis = sphBasisMap[basisNum];

		if (basis.layer != layer) 
			continue;

		double coef = ionCoefSphhar(basisNum, tmpobs, false);

		KFKey key;
		key.type	= KF::IONOSPHERIC;
		key.num		= basisNum;

		double val = 0;
		double std = 0;
		kfState.getKFValue(key, val, &std);

		iono	+= 		coef * val;
		var		+= SQR(	coef)* std;
	}

	return iono;
}

void ionOutputSphcal(
	Trace&		trace, 
	KFState&	kfState)
{
	SSRAtmGlobal atmGlob;
	atmGlob.numberLayers = acsConfig.ionModelOpts.layer_heights.size();
	for (int j = 0; j < atmGlob.numberLayers; j++)
	{
		atmGlob.layers[j].height	= acsConfig.ionModelOpts.layer_heights[j] / 1000;
		atmGlob.layers[j].maxOrder	= acsConfig.ionModelOpts.function_order;
		atmGlob.layers[j].maxDegree	= acsConfig.ionModelOpts.function_degree;
	}
	
	tracepdeex (4, trace, "\n#IONO_MODL  tow   indx hght order degr part     value       variance");
	for (auto [key, index] : kfState.kfIndexMap)
	{
		if (key.type != KF::IONOSPHERIC)
			continue;
		
		SphBasis& basis		= sphBasisMap[key.num];
		
		auto& ionoRecord	= atmGlob.layers[basis.layer].sphHarmonic[key.num];
		ionoRecord.layer	= basis.layer;
		ionoRecord.order	= basis.order;
		ionoRecord.degree	= basis.degree;
		ionoRecord.trigType	= basis.trigType;
		
		kfState.getKFValue(key, ionoRecord.value, &ionoRecord.variance);
		
		GTow tow = kfState.time;
		tracepdeex (4, trace, "\nIONO_MODL %6d %3d  %4.0f  %2d   %2d   %s  %10.4f %12.5e",
					(int)tow, 
					key.num, 
					atmGlob.layers[basis.layer].height, 
					basis.order, 
					basis.degree, 
					basis.trigType._to_string(), 
					ionoRecord.value, 
					sqrt(ionoRecord.variance));
	}
	tracepdeex (4, trace, "\n");
	
	atmGlob.time = kfState.time;
	
	nav.ssrAtm.atmosGlobalMap[kfState.time] = atmGlob;
}

bool getEpcSsrIono(
	GTime			time,		///< time of ionosphere correction
	SSRAtmGlobal&	atmGlob,	///< SSR atmospheric correction
	Vector3d&		rSat,		///< Satellite position
	Vector3d&		rRec,		///< receiver position
	double& 		iono,		///< Ionosphere delay (in TECu)
	double& 		var)		///< Ionosphere variance
{
	var		= 0;
	iono	= 0;
	
	if (fabs((time - atmGlob.time).to_double()) > acsConfig.ssrInOpts.global_vtec_valid_time)
		return false;
	
	if (sphBasisIndexMaps.size() < atmGlob.layers.size())
	{
		sphBasisIndexMaps.clear();
		int maxdeg = 0;
		int maxord = 0;
		for (auto& [hind,atmLay]: atmGlob.layers)
		{
			if (maxdeg < atmLay.maxDegree)		maxdeg = atmLay.maxDegree;
			if (maxord < atmLay.maxOrder)		maxord = atmLay.maxOrder;
			
			acsConfig.ionModelOpts.layer_heights[hind] = atmLay.height;
		}
		
		acsConfig.ionModelOpts.function_degree	= maxdeg;
		acsConfig.ionModelOpts.function_order	= maxord;
	
		configIonModelSphhar();
	}
	
	VectorPos pos = ecef2pos(rRec);
	
	Vector3d e;
	double r = geodist(rSat, rRec, e);
	double azel[2];
	satazel(pos, e, azel);
		
	for (auto& [layer, atmLay]: atmGlob.layers)
	{
		VectorPos posp;
		double slantFactor = ionppp(pos, azel, RE_WGS84 / 1000, atmLay.height, posp);
	
		if (ippCheckSphhar(time, posp) == false)
			return 0;
		
		GObs tmpobs;
		tmpobs.ippMap[layer].lat			= posp[0];
		tmpobs.ippMap[layer].lon			= posp[1];
		tmpobs.ippMap[layer].slantFactor	= slantFactor;
		
		for (auto& [ind, harmonic] : atmLay.sphHarmonic)
		{
			int reindex = sphBasisIndexMaps[harmonic.layer][harmonic.degree][harmonic.order][harmonic.trigType];
			
			double comp = ionCoefSphhar(reindex, tmpobs, true);
			
			iono	+=		comp	* harmonic.value;
			var		+= SQR(	comp)	* harmonic.variance;
		}
	}
	
	var += atmGlob.vtecQuality;
	
	return iono;	//todo aaron, converting to bool
}


bool getIGSSSRIono(
	GTime		time,	///< time of ionosphere correction
	SSRAtm&		ssrAtm,	///< SSR atmospheric correction
	Vector3d&	rSat,	///< Satellite position
	Vector3d&	rRec,	///< receiver position
	double& 	iono,	///< Ionosphere delay (in TECu)
	double& 	var)	///< Ionosphere variance
{
	var		= 0;
	iono	= 0;
	
	auto it = ssrAtm.atmosGlobalMap.lower_bound(time);
	if (it == ssrAtm.atmosGlobalMap.end())
		return false;

	auto& [t0, ssrAtm0] = *it;
	
	double iono0	= 0;
	double var0		= 0;
	bool pass0		= getEpcSsrIono(t0, ssrAtm0, rRec, rSat, iono0, var0);
	
	double iono1	= 0;
	double var1		= 0;
	bool pass1;
	double a;
	if (it == ssrAtm.atmosGlobalMap.begin())
	{
		pass1 = false;
	}
	else
	{
		it--;
		auto& [t1, ssrAtm1] = *it;
		pass1		= getEpcSsrIono(t1, ssrAtm1, rRec, rSat, iono1, var1);
		if (pass1)
		{
			a = (time - t0)/(t1 - t0);
		}
	}
	
	if (!pass0 && !pass1) {	var = -1;		return 0;		}
	if ( pass0 && !pass1) {	var = var0;		return iono0;	}
	if (!pass0 &&  pass1) {	var = var1;		return iono1;	}

	var = var0		* SQR(1-a)	+ var1	* SQR(a);
	return iono0	*	 (1-a)	+ iono1	*	  a;	//todo aaron, casting bad?
}
