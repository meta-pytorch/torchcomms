# MICPOWER Component

The MICPOWER component enables PAPI to access power readings reported on Intel MIC cards.

* [Enabling the MICPOWER Component](#enabling-the-micpower-component)
* [FAQ](#faq)

***
## Enabling the MICPOWER Component

To enable reading of MICPOWER counters the user needs to link against a PAPI library that was configured with the MICPOWER component enabled. As an example the following command: `./configure --with-components="MICPOWER"` is sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available to the user, and whether they are disabled, and when they are disabled why.

***
## FAQ

The values are reported in /sys/class/micras/power

	cat /sys/class/micras/power
	115000000
	113000000
	113000000
	129000000
	38000000
	29000000
	46000000
	0 0 1033000
	0 0 1501000
	0 0 1000000

This corresponds to the reading portions of the following MrRspPower structure. 

	typedef struct mr_rsp_pws {	/* Power status */
	  uint32_t	prr;				/* Current reading, in uW */
	  uint8_t p_val;                /* Valid bits, power */
	} MrRspPws;
	
	typedef struct mr_rsp_vrr {	/* Voltage regulator status */
	  uint32_t pwr;                 /* Power reading, in uW */
	  uint32_t cur;                 /* Current, in uA */
	  uint32_t volt;                /* Voltage, in uV */
	  uint8_t p_val;                /* Valid bits, power */
	  uint8_t c_val;                /* Valid bits, current */
	  uint8_t v_val;                /* Valid bits, voltage */
	} MrRspVrr;
	
	typedef struct mr_rsp_power {
	  MrRspPws tot0;                /* Total power, win 0 */
	  MrRspPws tot1;                /* Total power, win 1 */
	  MrRspPws	pcie;				/* PCI-E connector power */
	  MrRspPws	inst;				/* Instantaneous power */
	  MrRspPws	imax;				/* Max Instantaneous power */
	  MrRspPws	c2x3;				/* 2x3 connector power */
	  MrRspPws	c2x4;				/* 2x4 connector power */
	  MrRspVrr	vccp;				/* Core rail */
	  MrRspVrr	vddg;				/* Uncore rail */
	  MrRspVrr	vddq;				/* Memory subsystem rail */
	} MrRspPower;

