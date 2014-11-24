package ml.shifu.shifu.udf;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.KSIVCalculator;
import ml.shifu.shifu.core.StreamingBasicStatsCalculator;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.Accumulator;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;


public class StreamingCalculateStatesUDF extends AbstractTrainerUDF<Tuple> /*implements Accumulator<Tuple>*/ {

    private final static Logger log = LoggerFactory.getLogger(StreamingCalculateStatesUDF.class);

    //for numerical
    private List<Double> numericalBins = null;

    //for categorical
    private List<String> categoricalBins = null;

    //count
    private List<Integer> countNeg;
    private List<Integer> countPos;
    private List<Double> countWeightedNeg;
    private List<Double> countWeightedPos;

    //row count
    private Long rowCount = 0l;
    private Long missing = 0l;

    private final static TupleFactory tf = TupleFactory.getInstance();

    private static HashSet<String> positiveSet = new HashSet<String>();
    private static HashSet<String> negativeSet = new HashSet<String>();
    private ColumnConfig thisColumn = null;

    private StreamingBasicStatsCalculator basicStatsCalculator = new StreamingBasicStatsCalculator();
    private KSIVCalculator ksivCalculator = new KSIVCalculator();

    private Double valueThreshold = 1e6;

    public StreamingCalculateStatesUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        if (modelConfig.getNumericalValueThreshold() != null) {
            valueThreshold = modelConfig.getNumericalValueThreshold();
        }

        positiveSet.addAll(modelConfig.getPosTags());
        negativeSet.addAll(modelConfig.getNegTags());

        log.debug("Value Threshold: " + valueThreshold);
    }

    private List<Double> stringToList(String input) {
        List<Double> doubleList = new ArrayList<Double>();

        String[] binArray = StringUtils.split(input, StreamingBinningUDF.CATEGORY_VAL_SEPARATOR);

        for (String bin : binArray) {
            try {
                doubleList.add(Double.valueOf(bin));
            } catch (NumberFormatException e) {
                //do nothing first
                e.printStackTrace();

                log.warn("Fail to convert string into double in CalculateStatsUDF from {}", bin);
            }
        }

        return doubleList;
    }
    
    private List<String> stringToListString(String input) {
    	List<String> stringList = new ArrayList<String>();

        String[] binArray = StringUtils.split(input, StreamingBinningUDF.CATEGORY_VAL_SEPARATOR);

        for (String bin : binArray) {
            	stringList.add(bin);
        }
        return stringList;
    }

    //@Override
    public void accumulate(Tuple b) throws IOException {
        if (b == null) return;

        String binning = (String) b.get(0);
        DataBag databag = (DataBag) b.get(1);

        for (Tuple t : databag) {

            this.rowCount++;

            if (t.size() != 4) continue;

            Integer group = (Integer) t.get(0);

            if (thisColumn == null) {
                thisColumn = columnConfigList.get(group);
            }

            String rawValue = t.get(1).toString();

            if (StringUtils.isEmpty(rawValue)) {
                this.missing++;
                continue;
            }

            String target = (String) t.get(2);

            String weightValue = t.get(3).toString();

            switch (thisColumn.getColumnType()) {
                case N:
                    if (numericalBins == null) {
                        numericalBins = this.stringToList(binning); //CommonUtils.stringToDoubleList(binning);
                        log.info("the binning numbers is {}", numericalBins.toString());
                    }

                    if (countNeg == null) {
                        countNeg = new ArrayList<Integer>(Collections.nCopies(numericalBins.size(), 0));
                    }

                    if (countPos == null) {
                        countPos = new ArrayList<Integer>(Collections.nCopies(numericalBins.size(), 0));
                    }

                    try {
                        Double value = Double.valueOf(rawValue);

                        for (int i = 0; i < numericalBins.size() - 1; i++) {
                            if (value >= numericalBins.get(i) && value <= numericalBins.get(i + 1)) {
                                if (positiveSet.contains(target)) {
                                    countPos.set(i, countPos.get(i) + 1);
                                } else if (negativeSet.contains(target)) {
                                    countNeg.set(i, countNeg.get(i) + 1);
                                } else {
                                    log.warn("the {} is not presented in positive set or negative set, column name is {}",
                                            target,
                                            thisColumn.getColumnName());
                                    countNeg.set(i, countNeg.get(i) + 1);
                                }
                            }
                        }

                        basicStatsCalculator.aggregate(value);
                    } catch (NumberFormatException e) {
                        log.error("error in pasting string: {} to double", rawValue);
                    }
                    break;
                case C:
                    if (categoricalBins == null) {
                        categoricalBins = stringToListString(binning);
                    }

                    if (countNeg == null) {
                        countNeg = new ArrayList<Integer>(Collections.nCopies(categoricalBins.size(), 0));
                    }

                    if (countPos == null) {
                        countPos = new ArrayList<Integer>(Collections.nCopies(categoricalBins.size(), 0));
                    }

                    for (int i = 0; i < categoricalBins.size(); i++) {
                        if (rawValue.equals(categoricalBins.get(i))) {
                            if (positiveSet.contains(target)) {
                                countPos.set(i, countPos.get(i) + 1);
                            } else if (negativeSet.contains(target)) {
                                countNeg.set(i, countNeg.get(i) + 1);
                            } else {
                                log.warn("the {} is not presented in positive set or negative set, column name is {}",
                                        target,
                                        thisColumn.getColumnName());
                                countNeg.set(i, countNeg.get(i) + 1);
                            }
                        }
                    }

                    break;
                default:
                    break;
            }
        }
    }

    public List<Double> assemblyPositiveRate(List<Integer> countPos, List<Integer> countNeg) {

        if (countPos.size() != countNeg.size()) {
            throw new RuntimeException("The positive count is not equal to negative count");
        }

        List<Double> posRate = new ArrayList<Double>();

        for (int i = 0; i < countPos.size(); i++) {
            posRate.add((double) (countPos.get(i)) / (double) (countNeg.get(i) + countPos.get(i)));
        }

        return posRate;
    }

    //@Override
    public Tuple getValue() {

        Tuple t = tf.newTuple();

        if (thisColumn != null) {
            if (thisColumn.getColumnType().equals(ColumnConfig.ColumnType.N))
                t.append(this.numericalBins.toString());
            else
                t.append(this.categoricalBins.toString());
        }

        t.append(countNeg.toString());
        t.append(countPos.toString());
        t.append(null);
        t.append(assemblyPositiveRate(countPos, countNeg).toString());

        ksivCalculator.calculateKSIV(this.countPos, this.countNeg);

        t.append(ksivCalculator.getKS());
        t.append(ksivCalculator.getIV());

        basicStatsCalculator.complete();

        t.append(basicStatsCalculator.getMax());
        t.append(basicStatsCalculator.getMin());
        t.append(basicStatsCalculator.getMean());
        t.append(basicStatsCalculator.getStdDev());

        t.append(thisColumn.getColumnType().toString());

        t.append(basicStatsCalculator.getMedian());
        t.append(this.missing);
        t.append(this.rowCount);

        t.append(null);
        t.append(null);
        t.append(null);

        return t;
    }

    //@Override
    public void cleanup() {
        basicStatsCalculator = new StreamingBasicStatsCalculator();
        ksivCalculator = new KSIVCalculator();
        thisColumn = null;

        rowCount = 0l;
        missing = 0l;
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {
        cleanup();
        accumulate(input);
        return getValue();
    }
}
