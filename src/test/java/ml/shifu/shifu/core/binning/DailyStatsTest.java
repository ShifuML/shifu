package ml.shifu.shifu.core.binning;


import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.dtrain.StringUtils;
import ml.shifu.shifu.util.BinUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import org.apache.commons.io.IOUtils;
import org.junit.*;

import java.io.IOException;
import java.util.*;

public class DailyStatsTest {

    private static final double EPS = 1e-6;

    private ModelConfig modelConfig;

    private List<ColumnConfig> columnConfigList;

    private int tagColumnNum = -1;

    private int weightedColumnNum = -1;

    private boolean statsExcludeMissingValue = true;

    private Set<String> posTags;

    private Set<String> negTags;

    private Map<Integer, Map<String, List<CalItemInfo>>> dataset = new HashMap<>();

    @Test
    public void calculateDailyStats() throws IOException {
        loadData();
        calDailyStats();
    }

    private void loadData() throws IOException {
        this.modelConfig = CommonUtils.loadModelConfig(DailyStatsTest.class.getClassLoader().getResource("dailystats/ModelConfig.json").getPath(),
                RawSourceData.SourceType.LOCAL);
        this.columnConfigList = CommonUtils
                .loadColumnConfigList(DailyStatsTest.class.getClassLoader().getResource("dailystats/ColumnConfig.json").getPath(), RawSourceData.SourceType.LOCAL);
        loadWeightColumnNum();
        loadTagWeightNum();

        String allItems = IOUtils.toString(DailyStatsTest.class.getClassLoader().getResourceAsStream("dailystats/part-03"));
        String [] lines = StringUtils.split(allItems, "\n");
        for(String line : lines){
            String [] tokens = StringUtils.split(line, "|");
            for(int i = 0; i < tokens.length; i++){
                if(org.apache.commons.lang3.StringUtils.isEmpty(tokens[i])){
                    continue;
                }
                Map<String, List<CalItemInfo>> map = dataset.get(i);
                if(map == null){
                    map = new HashMap<>();
                    dataset.put(i, map);
                }
                List<CalItemInfo> list = map.get(tokens[tokens.length - 1]);
                if(list == null){
                    list = new ArrayList<>();
                    map.put(tokens[tokens.length - 1], list);
                }
                CalItemInfo info = new CalItemInfo();
                info.setVal(tokens[i]);
                info.setTag(tokens[tagColumnNum]);
                info.setWeight(getWeight(tokens));
                list.add(info);
            }
        }

        this.posTags = new HashSet<String>(modelConfig.getPosTags());
        this.negTags = new HashSet<String>(modelConfig.getNegTags());
    }

    private void calDailyStats(){
        for(Map.Entry<Integer, Map<String, List<CalItemInfo>>> entry : dataset.entrySet()){
            ColumnConfig columnConfig = this.columnConfigList.get(entry.getKey());
            for(Map.Entry<String, List<CalItemInfo>> dateEntry : entry.getValue().entrySet()){
                List<CalItemInfo> list = dateEntry.getValue();
                System.out.println(entry.getKey() + " " + dateEntry.getKey() + " "+list);
                DailyStatsInfo dailyStatsInfo = new DailyStatsInfo();
                dailyStatsInfo.setCount(list.size());

                for(CalItemInfo item : list){
                    boolean isValid = true;
                    double val = 0;


                    int binNum = 0;
                    int totalSize = 0;
                    if(columnConfig.isCategorical()) {
                        totalSize = columnConfig.getBinCategory().size();
                        binNum = quickLocateCategoricalBin(columnConfig.getBinCategory(), item.getVal());

                    }
                    else if(columnConfig.isNumerical()) {
                        totalSize = columnConfig.getBinBoundary().size();
                        binNum = getBinNum(columnConfig.getBinBoundary(), item.getVal());
                        try{
                            val = Double.parseDouble(item.getVal());
                        }catch(Exception e){
                            isValid = false;
                        }
                        if(org.apache.commons.lang3.StringUtils.isEmpty(item.getVal()) || modelConfig.getDataSet().getMissingOrInvalidValues().contains(item.getVal()) || !isValid){
                            dailyStatsInfo.setMissingCount(dailyStatsInfo.getMissingCount() + 1);
                        }
                        dailyStatsInfo.setSum(dailyStatsInfo.getSum() + val);
                        double squalVal = val * val;
                        dailyStatsInfo.setSquaredSum(dailyStatsInfo.getSquaredSum() + squalVal);
                        dailyStatsInfo.setTripleSum(dailyStatsInfo.getTripleSum() + squalVal * val);
                        dailyStatsInfo.setQuarticSum(dailyStatsInfo.getQuarticSum() + squalVal * squalVal);

                        if(Double.compare(dailyStatsInfo.getMax(), val) < 0) {
                            dailyStatsInfo.setMax(val);
                        }
                        if(Double.compare(dailyStatsInfo.getMin(), val) > 0) {
                            dailyStatsInfo.setMin(val);
                        }
                    }
                    if(dailyStatsInfo.getBinCountPos() == null){
                        dailyStatsInfo.setBinCountPos(new long[totalSize + 1]);
                    }
                    if(dailyStatsInfo.getBinWeightPos() == null){
                        dailyStatsInfo.setBinWeightPos(new double[totalSize + 1]);
                    }
                    if(dailyStatsInfo.getBinCountNeg() == null){
                        dailyStatsInfo.setBinCountNeg(new long[totalSize + 1]);
                    }
                    if(dailyStatsInfo.getBinWeightNeg() == null){
                        dailyStatsInfo.setBinWeightNeg(new double[totalSize + 1]);
                    }
                    if(dailyStatsInfo.getBinCountTotal() == null){
                        dailyStatsInfo.setBinCountTotal(new long[totalSize + 1]);
                    }

                    if(binNum > -1) {
                        if (modelConfig.isRegression()) {
                            if (posTags.contains(item.getTag())) {
                                dailyStatsInfo.getBinCountPos()[binNum] += 1L;
                                dailyStatsInfo.getBinWeightPos()[binNum] += item.getWeight();
                            } else if (negTags.contains(item.getTag())) {
                                dailyStatsInfo.getBinCountNeg()[binNum] += 1L;
                                dailyStatsInfo.getBinWeightNeg()[binNum] += item.getWeight();
                            }
                        } else {
                            // for multiple classification, set bin count to BinCountPos and leave BinCountNeg empty
                            dailyStatsInfo.getBinCountPos()[binNum] += 1L;
                            dailyStatsInfo.getBinWeightPos()[binNum] += item.getWeight();
                        }
                    }
                }

                for(int i = 0; i < dailyStatsInfo.getBinCountPos().length; i++){
                    dailyStatsInfo.getBinCountTotal()[i] += dailyStatsInfo.getBinCountPos() == null ? 0 : dailyStatsInfo.getBinCountPos()[i];
                    dailyStatsInfo.getBinCountTotal()[i] += dailyStatsInfo.getBinCountNeg() == null ? 0 : dailyStatsInfo.getBinCountNeg()[i];
                }

                if(columnConfig.isNumerical()) {
                    long p25Count = dailyStatsInfo.getCount() / 4;
                    long medianCount = p25Count * 2;
                    long p75Count = p25Count * 3;
                    int currentCount = 0;
                    for(int i = 0; i < columnConfig.getBinBoundary().size(); i++) {

                        double left = getCutoffBoundary(columnConfig.getBinBoundary().get(i), dailyStatsInfo.getMax(), dailyStatsInfo.getMin());
                        double right = ((i == columnConfig.getBinBoundary().size() - 1) ?
                                dailyStatsInfo.getMax() : getCutoffBoundary(columnConfig.getBinBoundary().get(i + 1), dailyStatsInfo.getMax(), dailyStatsInfo.getMin()));
                        if (p25Count >= currentCount && p25Count < currentCount + dailyStatsInfo.getBinCountTotal()[i]) {
                            dailyStatsInfo.setP25th(((p25Count - currentCount) / (double) dailyStatsInfo.getBinCountTotal()[i])
                                    * ( right - left) + left);
                        }

                        if (medianCount >= currentCount && medianCount < currentCount + dailyStatsInfo.getBinCountTotal()[i]) {
                            dailyStatsInfo.setMedian(((medianCount - currentCount) / (double) dailyStatsInfo.getBinCountTotal()[i])
                                    * ( right - left) + left);
                        }

                        if (p75Count >= currentCount && p75Count < currentCount + dailyStatsInfo.getBinCountTotal()[i]) {
                            dailyStatsInfo.setP75th(((p75Count - currentCount) / (double) dailyStatsInfo.getBinCountTotal()[i])
                                    * ( right - left) + left);
                            // when get 75 percentile stop it
                            break;
                        }
                        currentCount += dailyStatsInfo.getBinCountTotal()[i];
                    }
                }

                double[] binPosRate;
                if(modelConfig.isRegression()) {
                    binPosRate = computePosRate(dailyStatsInfo.getBinCountPos(), dailyStatsInfo.getBinCountNeg());
                } else {
                    // for multiple classfication, use rate of categories to compute a value
                    binPosRate = computeRateForMultiClassfication(dailyStatsInfo.getBinCountPos());
                }

                if(columnConfig.isCategorical()) {
                    dailyStatsInfo.setMin(Double.MAX_VALUE);
                    dailyStatsInfo.setMax(Double.MIN_VALUE);
                    dailyStatsInfo.setSum(0d);
                    dailyStatsInfo.setSquaredSum(0d);
                    for (int i = 0; i < binPosRate.length; i++) {
                        if (!Double.isNaN(binPosRate[i])) {
                            if (Double.compare(dailyStatsInfo.getMax(), binPosRate[i]) < 0) {
                                dailyStatsInfo.setMax(binPosRate[i]);
                            }

                            if (Double.compare(dailyStatsInfo.getMin(), binPosRate[i]) > 0) {
                                dailyStatsInfo.setMin(binPosRate[i]);
                            }
                            long binCount = dailyStatsInfo.getBinCountPos()[i] + dailyStatsInfo.getBinCountNeg()[i];
                            dailyStatsInfo.setSum(dailyStatsInfo.getSum() + binPosRate[i] * binCount);
                            double squaredVal = binPosRate[i] * binPosRate[i];
                            dailyStatsInfo.setSquaredSum(dailyStatsInfo.getSquaredSum() + squaredVal * binCount);
                            dailyStatsInfo.setTripleSum(dailyStatsInfo.getTripleSum() + squaredVal * binPosRate[i] * binCount);
                            dailyStatsInfo.setQuarticSum(dailyStatsInfo.getQuarticSum() + squaredVal * squaredVal * binCount);
                        }
                    }
                }

                long realCount = this.statsExcludeMissingValue ? (dailyStatsInfo.getCount() - dailyStatsInfo.getMissingCount()) : dailyStatsInfo.getCount();

                dailyStatsInfo.setMean(dailyStatsInfo.getSum() / realCount);

                dailyStatsInfo.setStdDev(Math.sqrt(Math.abs((dailyStatsInfo.getSquaredSum() - (dailyStatsInfo.getSum() * dailyStatsInfo.getSum()) / realCount + EPS) / (realCount - 1))));
                dailyStatsInfo.setaStdDev(Math.sqrt(Math.abs((dailyStatsInfo.getSquaredSum() - (dailyStatsInfo.getSum() * dailyStatsInfo.getSum()) / realCount + EPS) / realCount)));

                dailyStatsInfo.setSkewness(ColumnStatsCalculator.computeSkewness(realCount, dailyStatsInfo.getMean(), dailyStatsInfo.getaStdDev(), dailyStatsInfo.getSum(), dailyStatsInfo.getSquaredSum(), dailyStatsInfo.getTripleSum()));
                dailyStatsInfo.setKurtosis(ColumnStatsCalculator.computeKurtosis(realCount, dailyStatsInfo.getMean(), dailyStatsInfo.getaStdDev(), dailyStatsInfo.getSum(), dailyStatsInfo.getSquaredSum(), dailyStatsInfo.getTripleSum(),
                        dailyStatsInfo.getQuarticSum()));

                if(modelConfig.isRegression()) {
                    dailyStatsInfo.setColumnCountMetrics(ColumnStatsCalculator.calculateColumnMetrics(dailyStatsInfo.getBinCountNeg(), dailyStatsInfo.getBinCountPos()));
                    dailyStatsInfo.setColumnWeightMetrics(ColumnStatsCalculator.calculateColumnMetrics(dailyStatsInfo.getBinWeightNeg(), dailyStatsInfo.getBinWeightPos()));
                }

                System.out.println(dailyStatsInfo);
            }
        }
    }


    private double[] computePosRate(long[] binCountPos, long[] binCountNeg) {
        assert binCountPos != null && binCountNeg != null && binCountPos.length == binCountNeg.length;
        double[] posRate = new double[binCountPos.length];
        for(int i = 0; i < posRate.length; i++) {
            if(Double.compare(binCountPos[i] + binCountNeg[i], 0d) != 0) {
                // only compute effective pos rate, if /0, don't do it
                posRate[i] = binCountPos[i] * 1.0d / (binCountPos[i] + binCountNeg[i]);
            }
        }
        return posRate;
    }

    private double[] computeRateForMultiClassfication(long[] binCount) {
        double[] rate = new double[binCount.length];
        double sum = 0d;
        for(int i = 0; i < binCount.length; i++) {
            sum += binCount[i];
        }
        for(int i = 0; i < binCount.length; i++) {
            if(Double.compare(sum, 0d) != 0) {
                rate[i] = binCount[i] * 1.0d / sum;
            }
        }
        return rate;
    }

    private void loadWeightColumnNum() {
        String weightColumnName = this.modelConfig.getDataSet().getWeightColumnName();
        if(weightColumnName != null && weightColumnName.length() != 0) {
            for(int i = 0; i < this.columnConfigList.size(); i++) {
                ColumnConfig config = this.columnConfigList.get(i);
                if(config.getColumnName().equals(weightColumnName)) {
                    this.weightedColumnNum = i;
                    break;
                }
            }
        }
    }

    private void loadTagWeightNum() {
        for(ColumnConfig config: this.columnConfigList) {
            if(config.isTarget()) {
                this.tagColumnNum = config.getColumnNum();
                break;
            }
        }

        if(this.tagColumnNum == -1) {
            throw new RuntimeException("No valid target column.");
        }
    }

    private double getWeight(String[] tokens){
        Double weight = 1.0;
        try {
            weight = (this.weightedColumnNum == -1 ? 1.0d : Double.valueOf(tokens[weightedColumnNum]));
            if(weight < 0) {
                throw new IllegalStateException(
                        "Please check weight column in eval, exceptional weight count is over 5000");
            }
        } catch (NumberFormatException e) {
            throw new IllegalStateException(
                    "Please check weight column in eval, exceptional weight count is over 5000");
        }
        return weight;
    }


    private int quickLocateCategoricalBin(List<String> list, String val) {
        for (int i = 0; i < list.size(); i++){
            if(org.apache.commons.lang.StringUtils.equals(list.get(i), val)){
                return i;
            }
        }
        return -1;
    }

    public static int getBinNum(List<Double> binBoundaryList, String columnVal) {
        if(org.apache.commons.lang.StringUtils.isBlank(columnVal)) {
            return -1;
        }
        double dval = 0.0;
        try {
            dval = Double.parseDouble(columnVal);
        } catch (Exception e) {
            return -1;
        }
        return BinUtils.getBinIndex(binBoundaryList, dval);
    }

    private double getCutoffBoundary(double val, double max, double min) {
        if ( val == Double.POSITIVE_INFINITY ) {
            return max;
        } else if ( val == Double.NEGATIVE_INFINITY ) {
            return min;
        } else {
            return val;
        }
    }
}

class CalItemInfo{

    private String val;

    private double weight;

    private String tag;

    public String getVal() {
        return val;
    }

    public void setVal(String val) {
        this.val = val;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public String getTag() {
        return tag;
    }

    public void setTag(String tag) {
        this.tag = tag;
    }

    @Override
    public String toString() {
        return "CalItemInfo{" +
                "val='" + val + '\'' +
                ", weight=" + weight +
                ", tag='" + tag + '\'' +
                '}';
    }
}

class DailyStatsInfo{

    private double min = Double.MAX_VALUE;

    private double max = Double.MIN_VALUE;

    private int count = 0;

    private int missingCount = 0;

    private double sum = 0.0d;

    private double squaredSum = 0.0d;

    private double tripleSum = 0.0d;

    private double quarticSum = 0.0d;

    private double p25th = 0.0d;

    private double p75th = 0.0d;

    private double median = 0.0d;

    private double mean = 0.0d;

    private double stdDev = 0.0d;

    private double aStdDev = 0.0d;

    private double skewness = 0.0d;

    private double kurtosis = 0.0d;

    private long[] binCountPos;

    private long[] binCountNeg;

    private double[] binWeightPos;

    private double[] binWeightNeg;

    private long[] binCountTotal;

    private ColumnStatsCalculator.ColumnMetrics columnCountMetrics = null;

    private ColumnStatsCalculator.ColumnMetrics columnWeightMetrics = null;

    public double getMean() {
        return mean;
    }

    public void setMean(double mean) {
        this.mean = mean;
    }

    public double getMin() {
        return min;
    }

    public void setMin(double min) {
        this.min = min;
    }

    public double getMax() {
        return max;
    }

    public void setMax(double max) {
        this.max = max;
    }

    public int getCount() {
        return count;
    }

    public void setCount(int count) {
        this.count = count;
    }

    public int getMissingCount() {
        return missingCount;
    }

    public void setMissingCount(int missingCount) {
        this.missingCount = missingCount;
    }

    public double getSum() {
        return sum;
    }

    public void setSum(double sum) {
        this.sum = sum;
    }

    public double getSquaredSum() {
        return squaredSum;
    }

    public void setSquaredSum(double squaredSum) {
        this.squaredSum = squaredSum;
    }

    public double getTripleSum() {
        return tripleSum;
    }

    public void setTripleSum(double tripleSum) {
        this.tripleSum = tripleSum;
    }

    public double getQuarticSum() {
        return quarticSum;
    }

    public void setQuarticSum(double quarticSum) {
        this.quarticSum = quarticSum;
    }

    public double getP25th() {
        return p25th;
    }

    public void setP25th(double p25th) {
        this.p25th = p25th;
    }

    public double getP75th() {
        return p75th;
    }

    public void setP75th(double p75th) {
        this.p75th = p75th;
    }

    public double getMedian() {
        return median;
    }

    public void setMedian(double median) {
        this.median = median;
    }

    public double getStdDev() {
        return stdDev;
    }

    public void setStdDev(double stdDev) {
        this.stdDev = stdDev;
    }

    public double getaStdDev() {
        return aStdDev;
    }

    public void setaStdDev(double aStdDev) {
        this.aStdDev = aStdDev;
    }

    public double getSkewness() {
        return skewness;
    }

    public void setSkewness(double skewness) {
        this.skewness = skewness;
    }

    public double getKurtosis() {
        return kurtosis;
    }

    public void setKurtosis(double kurtosis) {
        this.kurtosis = kurtosis;
    }

    public long[] getBinCountPos() {
        return binCountPos;
    }

    public void setBinCountPos(long[] binCountPos) {
        this.binCountPos = binCountPos;
    }

    public long[] getBinCountNeg() {
        return binCountNeg;
    }

    public void setBinCountNeg(long[] binCountNeg) {
        this.binCountNeg = binCountNeg;
    }

    public double[] getBinWeightPos() {
        return binWeightPos;
    }

    public void setBinWeightPos(double[] binWeightPos) {
        this.binWeightPos = binWeightPos;
    }

    public double[] getBinWeightNeg() {
        return binWeightNeg;
    }

    public void setBinWeightNeg(double[] binWeightNeg) {
        this.binWeightNeg = binWeightNeg;
    }

    public long[] getBinCountTotal() {
        return binCountTotal;
    }

    public void setBinCountTotal(long[] binCountTotal) {
        this.binCountTotal = binCountTotal;
    }

    public ColumnStatsCalculator.ColumnMetrics getColumnCountMetrics() {
        return columnCountMetrics;
    }

    public void setColumnCountMetrics(ColumnStatsCalculator.ColumnMetrics columnCountMetrics) {
        this.columnCountMetrics = columnCountMetrics;
    }

    public ColumnStatsCalculator.ColumnMetrics getColumnWeightMetrics() {
        return columnWeightMetrics;
    }

    public void setColumnWeightMetrics(ColumnStatsCalculator.ColumnMetrics columnWeightMetrics) {
        this.columnWeightMetrics = columnWeightMetrics;
    }

    @Override
    public String toString() {
        return "DailyStatsInfo{" +
                "min=" + min +
                ", max=" + max +
                ", count=" + count +
                ", missingCount=" + missingCount +
                ", sum=" + sum +
                ", squaredSum=" + squaredSum +
                ", tripleSum=" + tripleSum +
                ", quarticSum=" + quarticSum +
                ", p25th=" + p25th +
                ", p75th=" + p75th +
                ", median=" + median +
                ", mean=" + mean +
                ", stdDev=" + stdDev +
                ", aStdDev=" + aStdDev +
                ", skewness=" + skewness +
                ", kurtosis=" + kurtosis +
                ", binCountPos=" + Arrays.toString(binCountPos) +
                ", binCountNeg=" + Arrays.toString(binCountNeg) +
                ", binWeightPos=" + Arrays.toString(binWeightPos) +
                ", binWeightNeg=" + Arrays.toString(binWeightNeg) +
                ", binCountTotal=" + Arrays.toString(binCountTotal) +
                ", columnCountMetrics=" + columnCountMetrics +
                ", columnWeightMetrics=" + columnWeightMetrics +
                '}';
    }
}
