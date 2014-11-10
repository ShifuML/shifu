package ml.shifu.shifu.udf;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.core.Binning;
import ml.shifu.shifu.core.Estimator;
import org.apache.pig.Accumulator;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


public class StreamingBinningUDF extends AbstractTrainerUDF<String> implements Accumulator<String> {

    private final static Logger log = LoggerFactory.getLogger(Binning.class);
    private int maxBinSize = 0;

    //for equal positive
    private static HashSet<String> positiveSet = new HashSet<String>();

    //for equal interval
    private Double max = Double.MIN_VALUE;
    private Double min = Double.MAX_VALUE;

    //for categorical
    private Set<String> categories = new HashSet<String>();

    private Estimator<Double> estimator;
    private ColumnConfig thisColumn = null;

    public StreamingBinningUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathColumnConfig, pathModelConfig);

        positiveSet.addAll(modelConfig.getPosTags());

        maxBinSize = modelConfig.getStats().getMaxNumBin();

        estimator = new Estimator<Double>(maxBinSize);

    }

    public void accumulate(Tuple b) throws IOException {

        if (b == null || b.size() == 0) return;

        DataBag bag = (DataBag) b.get(0);

        for (Tuple t : bag) {

            if (t.size() != 4) continue;

            Integer group = (Integer) t.get(0);

            if (thisColumn == null) {
                thisColumn = columnConfigList.get(group);
            }

            String rawValue = t.get(1).toString();

            String target = (String) t.get(2);

            String weightValue = t.get(3).toString();

            // categorical or numerical

            switch (thisColumn.getColumnType()){
                case N :
                    Double value = 0.0;
                    try {
                        value = Double.valueOf(rawValue);
                    } catch (NumberFormatException e) {
                        log.warn("Fail to handle column: {} {}  expected is numerical, but could not cast to double",
                                thisColumn.getColumnNum() ,
                                thisColumn.getColumnName()
                        );
                        continue;
                    }

                    switch (modelConfig.getBinningMethod()) {
                        case EqualPositive:
                            if (positiveSet.contains(target)) {
                                estimator.add(value);
                            }
                            break;
                        case EqualTotal:
                            estimator.add(value);
                            break;
                        case EqualInterval:
                            if (max < value) {
                                this.max = value;
                            }

                            if (min > value) {
                                this.min = value;
                            }
                            break;
                        default:
                            break;
                    }
                    break;
                case C :
                    this.categories.add(rawValue);
                    break;
                default :
                    break;
            }
        }

    }

    public void cleanup() {
        this.estimator.clear();
        this.min = Double.MAX_VALUE;
        this.max = Double.MIN_VALUE;
    }

    public String getValue() {

        if(thisColumn == null) return new ArrayList<Double>().toString();

        if (thisColumn.getColumnType().equals(ColumnConfig.ColumnType.N) &&
                !modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualInterval)) {
            //numeric binning
            return binMerge(estimator.getBin());
        } else if (thisColumn.getColumnType().equals(ColumnConfig.ColumnType.N) &&
                modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualInterval)){
            //equal binning
            return getEqualIntervalBinning(this.min, this.max, this.maxBinSize);
        } else if (thisColumn.getColumnType().equals(ColumnConfig.ColumnType.C)){
            return this.categories.toString();
        } else {
            return null;
        }
    }

    public String binMerge(List<Double> bins) {

        Double lastBin = Double.NEGATIVE_INFINITY;
        Double thisBin;

        List<Double> newbins = new ArrayList<Double>();

        for ( int i = 0 ; i < bins.size(); i++ ){
            thisBin = bins.get(i);
            if (Math.abs(lastBin - thisBin) > 1e-6) {
                newbins.add(thisBin);
            }

            lastBin = thisBin;
        }

        return newbins.toString();
    }

    public String getEqualIntervalBinning(double min, double max, int maxBinSize){

        List<Double> bins = new ArrayList<Double>(maxBinSize + 1);

        bins.add(min);

        if (Math.abs(min - max) < 1e-8) {
            bins.add(max);
            return bins.toString();
        }

        double interval = (max - min) / maxBinSize;

        for (int i = 1 ; i < maxBinSize; i ++) {

            double next = min + i * interval;
            double curr = bins.get(i - 1);

            if(Math.abs(curr - next) > 1e-8) {
                bins.add(next);
            }
        }

        bins.add(max);

        return bins.toString();
    }

    @Override
    public String exec(Tuple input) throws IOException {
        cleanup();
        accumulate(input);
        return getValue();
    }
}
