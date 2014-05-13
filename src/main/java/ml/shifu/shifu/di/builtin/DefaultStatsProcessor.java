package ml.shifu.shifu.di.builtin;


import com.google.inject.Inject;
import ml.shifu.shifu.container.CategoricalValueObject;
import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.container.obj.*;
import ml.shifu.shifu.di.spi.*;
import ml.shifu.shifu.util.CommonUtils;

import java.util.List;
import java.util.Map;

public class DefaultStatsProcessor implements StatsProcessor {

    private ColumnRawStatsCalculator rawStatsCalculator;
    private ColumnNumBinningCalculator numBinningCalculator;
    private ColumnCatBinningCalculator catBinningCalculator;
    private ColumnNumStatsCalculator numStatsCalculator;
    private ColumnBinStatsCalculator binStatsCalculator;



    private List<String> posTags;
    private List<String> negTags;
    private Integer numBins;


    @Inject
    public DefaultStatsProcessor(ColumnRawStatsCalculator rawStatsCalculator,
                                 ColumnNumBinningCalculator numBinningCalculator,
                                 ColumnCatBinningCalculator catBinningCalculator,
                                 ColumnNumStatsCalculator numStatsCalculator,
                                 ColumnBinStatsCalculator binStatsCalculator) {

        this.rawStatsCalculator = rawStatsCalculator;
        this.numBinningCalculator = numBinningCalculator;
        this.catBinningCalculator = catBinningCalculator;
        this.numStatsCalculator = numStatsCalculator;
        this.binStatsCalculator = binStatsCalculator;
    };


    public void process(ColumnConfig columnConfig, List<RawValueObject> rvoList) {


        columnConfig.setColumnRawStatsResult(rawStatsCalculator.calculate(rvoList, posTags, negTags));

        //TODO: Let user choose if column should be treated as Numerical or Categorical if not predefined in ColumnConfig

        ColumnBinningResult binningResult;
        ColumnNumStatsResult numStatsResult;

        if (columnConfig.isNumerical()) {
            List<NumericalValueObject> nvoList = CommonUtils.convertListRaw2Numerical(rvoList, posTags, negTags);
            binningResult = numBinningCalculator.calculate(nvoList, numBins);
            numStatsResult = numStatsCalculator.calculate(nvoList);
        } else {
            List<CategoricalValueObject> cvoList = CommonUtils.convertListRaw2Categorical(rvoList, posTags, negTags);
            binningResult = catBinningCalculator.calculate(cvoList);
            numStatsResult = numStatsCalculator.calculate(CommonUtils.convertListCategorical2Numerical(cvoList, binningResult));
        }
        columnConfig.setColumnBinningResult(binningResult);
        columnConfig.setColumnNumStatsResult(numStatsResult);

        columnConfig.setColumnBinStatsResult(binStatsCalculator.calculate(binningResult));
    }

    public void setParams(Map<String, Object> params) {
        if (params.containsKey("posTags")) {
            this.posTags = (List<String>) params.get("posTags");
        } else {
            throw new RuntimeException(this.getClass().getSimpleName() + " needs param: " + "posTags");
        }

        if (params.containsKey("negTags")) {
            this.negTags = (List<String>) params.get("negTags");
        } else {
            throw new RuntimeException(this.getClass().getSimpleName() + " needs param: " + "negTags");
        }

        if (params.containsKey("numBins")) {
            this.numBins = (Integer) params.get("numBins");
        } else {
            throw new RuntimeException(this.getClass().getSimpleName() + " needs param: " + "posTags");
        }

    }
}
