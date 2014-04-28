/**
 * Copyright [2012-2014] eBay Software Foundation
 *  
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *  
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core;

import ml.shifu.shifu.container.obj.ColumnConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *  Normalizer
 *  </p>
 *  formula:
 *  </p>
 *  <code>norm_result = (value - means) / stdev</code>
 *  </p>
 *  
 *  The stdDevCutOff should be setting, by default it's 4
 *  </p>
 *  
 *  The <code>value</code> should less than  mean + stdDevCutOff * stdev
 *  </p>
 *  and larger than mean - stdDevCutOff * stdev
 */
public class Normalizer {
	
	private static Logger log = LoggerFactory.getLogger(Normalizer.class);
	public static final double STD_DEV_CUTOFF = 4.0d;
	
	public enum NormalizeMethod {
	    ZScore, MaxMin;
	}

	private ColumnConfig config;
	private Double stdDevCutOff = 4.0;
	private NormalizeMethod method;

	/**
	 * Create @Normalizer, according @ColumnConfig
	 *     NormalizeMethod method will be NormalizeMethod.ZScore
	 *     stdDevCutOff will be STD_DEV_CUTOFF
	 * @param config - @ColumnConfig to create normalizer
	 */
    public Normalizer(ColumnConfig config) {
        this(config, NormalizeMethod.ZScore, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according @ColumnConfig and NormalizeMethod
     *     stdDevCutOff will be STD_DEV_CUTOFF
     * @param config - @ColumnConfig to create normalizer
     * @param method - NormalizMethod to use
     */
    public Normalizer(ColumnConfig config, NormalizeMethod method) {
        this(config, method, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according @ColumnConfig and NormalizeMethod
     *     NormalizeMethod method will be NormalizeMethod.ZScore
     * @param config - @ColumnConfig to create normalizer
     * @param cutoff - stand_dev_cutoff to use
     */
    public Normalizer(ColumnConfig config, Double cutoff) {
        this(config, NormalizeMethod.ZScore, STD_DEV_CUTOFF);
    }

    /**
     * Create @Normalizer, according @ColumnConfig and NormalizeMethod
     *     NormalizeMethod method will be NormalizeMethod.ZScore
     * @param config - @ColumnConfig to create normalizer
     * @param method - NormalizMethod to use
     * @param cutoff - stand_dev_cutoff to use
     */
    public Normalizer(ColumnConfig config, NormalizeMethod method, Double cutoff) {
        this.config = config;
        this.method = method;
        this.stdDevCutOff = cutoff;
    }
	
    /**
     * Normalize the input data for column
     * @param raw
     * @return
     */
    public Double normalize(String raw) {
        switch (method) {
        case ZScore:
            return getZScore(config, raw, stdDevCutOff);
        case MaxMin:
            return getMaxMinScore(config, raw);
        default:
            return 0.0;
        }
    }

    /**
     * Normalize the raw file, according the @ColumnConfig info
     * @param config - @ColumnConfig to normalize data
     * @param raw - raw input data 
     * @return - normalized value 
     */
	public static Double normalize(ColumnConfig config, String raw) {
	    return normalize(config, raw, NormalizeMethod.ZScore);
	}
	
	/**
	 * Normalize the raw file, according the @ColumnConfig info and normalized method
	 * @param config - @ColumnConfig to normalize data
	 * @param raw - raw input data 
	 * @param method - the method used to do normalization
	 * @return - normalized value 
	 */
    public static Double normalize(ColumnConfig config, String raw, NormalizeMethod method) {
        return normalize(config, raw, method, STD_DEV_CUTOFF);
    }

    /**
     * Normalize the raw file, according the @ColumnConfig info and standard deviation cutoff
     * @param config - @ColumnConfig to normalize data
     * @param raw - raw input data 
     * @param stdDevCutoff - the standard deviation cutoff to use
     * @return - normalized value
     */
    public static Double normalize(ColumnConfig config, String raw, double stdDevCutoff) {
        return normalize(config, raw, NormalizeMethod.ZScore, stdDevCutoff);
    }
    
    /**
     * Normalize the raw file, according the @ColumnConfig info, normalized method and standard deviation cutoff
     * @param config - @ColumnConfig to normalize data
     * @param raw - raw input data 
     * @param method - the method used to do normalization
     * @param stdDevCutoff - the standard deviation cutoff to use
     * @return - normalized value
     */
    public static Double normalize(ColumnConfig config, String raw, NormalizeMethod method, double stdDevCutoff) {
        if ( method == null ) {
            method = NormalizeMethod.ZScore;
        }
        
        switch(method){
        case ZScore:
            return getZScore(config, raw, stdDevCutoff);
        case MaxMin:
            return getMaxMinScore(config, raw);
        default:
            return 0.0;
        }
    }

    /**
     * Compute the normalized data for @NormalizeMethod.MaxMin
     * @param config - @ColumnConfig info
     * @param raw - input column value
     * @return - normalized value for MaxMin method
     */
    private static Double getMaxMinScore(ColumnConfig config, String raw) {
		if (config.isCategorical()) {
			//TODO, doesn't support
		} else {
			Double value = Double.parseDouble(raw);
			return (value - config.getColumnStats().getMin()) / 
			        (config.getColumnStats().getMax() - config.getColumnStats().getMin());
		}
		return null;
	}

    /**
     * Compute the normalized data for @NormalizeMethod.Zscore
     * @param config - @ColumnConfig info
     * @param raw - input column value
     * @param cutoff 
     * @return - normalized value for MaxMin method
     */
	private static Double getZScore(ColumnConfig config, String raw, Double cutoff) {
        Double stdDevCutOff;
        if (cutoff != null && !cutoff.isInfinite() && !cutoff.isNaN()) {
            stdDevCutOff = cutoff;
        } else {
            stdDevCutOff = STD_DEV_CUTOFF;
        }
        
        if (config.isCategorical()) {
            int index = config.getBinCategory().indexOf(raw);
            // TODO: use default. Not 0 !!!
            // Using the most frequent categorical value?
            if (index == -1) {
                return 0.0;
            } else {
                return computeZScore(config.getBinPosRate().get(index), config.getMean(), config.getStdDev(), stdDevCutOff);
            }
        } else {
            double value = 0.0;
            try {
                value = Double.parseDouble(raw);
            } catch (Exception e) {
            	log.debug("Not decimal format " + raw + ", using default!");
            	value = ((config.getMean() == null) ? 0.0 : config.getMean());
            }
            
            return computeZScore(value, config.getMean(), config.getStdDev(), stdDevCutOff);
        }
    }
	
	/**
	 * Compute the zscore, by original value, mean, standard deviation and standard deviation cutoff
	 * @param var - original value
	 * @param mean - mean value
	 * @param stdDev - standard deviation
	 * @param stdDevCutOff - standard deviation cutoff
	 * @return zscore
	 */
	public static double computeZScore(double var, double mean, double stdDev, double stdDevCutOff) {
        double maxCutOff = mean + stdDevCutOff * stdDev;
        if (var > maxCutOff) {
            var = maxCutOff;
        }
        
        double minCutOff = mean - stdDevCutOff * stdDev;
        if (var < minCutOff) {
            var = minCutOff;
        }

        if ( stdDev > 0.00001 ) {
            return ( var - mean ) / stdDev;
        } else {
            return 0.0;
        }
    }
}
