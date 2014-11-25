package ml.shifu.shifu.core.binning;

import java.util.ArrayList;
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
		
		Double dval = Double.valueOf(0.0);
        
        try {
            dval = Double.parseDouble(fval);
            estimator.add(dval);
        } catch (NumberFormatException e) {
            super.incInvalidValCnt();
        }
	}
	
	/**
	 * add negative infinity and positive infinity, merge same bins
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
		
		newBins.add(Double.NEGATIVE_INFINITY);
		Double cur = bins.get(0);
		newBins.add(cur);
		
		int i = 1;
		while (i < bins.size()) {
			if (Math.abs(cur - bins.get(i)) > 1e-4) {
				newBins.add(bins.get(i));
			}
			cur = bins.get(i);
			i ++;
		}		
		
		newBins.add(Double.POSITIVE_INFINITY);
		return newBins;		
	}

	@Override
	public List<Double> getDataBin() {
		return binMerge(estimator.getQuantiles());
	}

}
