package ml.shifu.shifu.core.binning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang.StringUtils;

import ml.shifu.shifu.core.MunroPatEstimator;

public class MunroPatBinning extends AbstractBinning<Double> {

	private MunroPatEstimator<Double> estimator = null;
	
	public MunroPatBinning(int binningNum, List<String> missingValList) {
		super(binningNum, missingValList);
		estimator = new MunroPatEstimator<Double>(binningNum);
	}

	public MunroPatBinning(int binningNum) {
		this(binningNum, null);
	}

	@Override
	public void addData(String val) {
		String fval = StringUtils.trimToEmpty(val);
		try {
        	Double dval = Double.valueOf(0.0);
            dval = Double.parseDouble(fval);
            estimator.add(dval);
            
        } catch (NumberFormatException e) {
            super.incInvalidValCnt();
        }
	}
	
	/**
	 * set min/max, merge same bins
	 * 
	 * @param bins
	 * @return
	 */
	private List<Double> binMerge(List<Double> bins){
		
		List<Double> newBins = new ArrayList<Double>();
		if (bins.size() == 0) {
			bins.add(Double.NaN);
			return bins;
		}
		
		Double cur = bins.get(0);
		newBins.add(cur);
		
		int i = 1;
		while (i < bins.size()) {
			if (Math.abs(cur - bins.get(i)) > 1e-10) {
				newBins.add(bins.get(i));
			}
			cur = bins.get(i);
			i ++;
		}
		
		if (newBins.size() == 1) {
			//special case since there is only 1 candidate in the bins
			double val = newBins.get(0);
			newBins = Arrays.asList(new Double[] {Double.NEGATIVE_INFINITY, val});
		} else if (newBins.size() == 2) {
			newBins.set(0, Double.NEGATIVE_INFINITY);
		} else {
			newBins.set(0, Double.NEGATIVE_INFINITY);
			//remove the max, and became open interval
			newBins.remove(newBins.size() - 1);
		}
		return newBins;		
	}

	@Override
	public List<Double> getDataBin() {
		return binMerge(estimator.getQuantiles());
	}

    public List<Double> getUnMergedDataBin(){
        return estimator.getQuantiles();
    }

}
