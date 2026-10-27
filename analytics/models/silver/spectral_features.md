{% docs fan_amp_bpfo_mean_5m %}
5-minute mean of fan_amp_bpfo (mm/s): the spectral amplitude at the fan bearing's outer-race
defect frequency (BPFO), the signature of a bearing defect. NULL when the pulse has no HF frame
or no frame of the window carries a usable value.
{% enddocs %}

{% docs fan_kurtosis_mean_5m %}
5-minute mean of fan_kurtosis: Pearson kurtosis of the fan velocity waveform (tones and Gaussian
noise stay at or below 3; the impacts of a bearing defect raise it). NULL without HF data.
{% enddocs %}

{% docs fan_crest_factor_mean_5m %}
5-minute mean of fan_crest_factor: peak / RMS of the fan velocity waveform (impulsiveness).
NULL without HF data.
{% enddocs %}

{% docs fan_amp_1x_n1_mean_5m %}
5-minute mean of fan_amp_1x_n1 (mm/s): the spectral amplitude at the fan shaft frequency
(fan unbalance). NULL without HF data.
{% enddocs %}

{% docs core_amp_1x_n2_mean_5m %}
5-minute mean of core_amp_1x_n2 (mm/s): the turbine rear frame's spectral amplitude at the core
shaft frequency (core unbalance; grows with HPT blade erosion and compressor fouling). NULL
without HF data.
{% enddocs %}

{% docs fan_band_energy_hf_mean_5m %}
5-minute mean of fan_band_energy_hf: the fraction (0..1) of the fan spectral energy above a
quarter of the sample rate, where defect impacts excite structural resonances. NULL without HF
data.
{% enddocs %}

{% docs oil_temp_c_mean_5m %}
5-minute mean of the oil temperature (process parameter OIL_T_A, degC; rises with wear). NULL
without HF data or when the frames do not carry the parameter.
{% enddocs %}

{% docs oil_quantity_mean_5m %}
5-minute mean of the oil quantity (process parameter OIL_QTY_A, L; drops in an oil-system
leak). NULL without HF data or when the frames do not carry the parameter.
{% enddocs %}

{% docs oil_filter_dp_mean_5m %}
5-minute mean of the oil filter differential pressure (process parameter OIL_DP_A, psid; rises
with bearing debris). NULL without HF data or when the frames do not carry the parameter.
{% enddocs %}

{% docs egt_ab_split_mean_5m %}
5-minute mean of egt_ab_split (degC): the disagreement of the redundant EGT sensors A and B, a
sensor-health signal. NULL without HF data.
{% enddocs %}
