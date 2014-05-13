/**
 * Copyright [2012-2013] eBay Software Foundation
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
package ml.shifu.shifu.udf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.container.CategoricalValueObject;
import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.container.obj.*;
import ml.shifu.shifu.di.module.StatsModule;
import ml.shifu.shifu.di.service.ColumnBinStatsService;
import ml.shifu.shifu.di.service.ColumnNumStatsService;
import ml.shifu.shifu.di.service.ColumnBinningService;
import ml.shifu.shifu.di.service.ColumnRawStatsService;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;

/**
 *
 * CalculateStatsUDF class is calculate the stats for each column
 *
 * Input: (columnNum, {(value, tag, weight), (value, tag, weight)...})
 *
 */
public class CalculateStatsUDF extends AbstractTrainerUDF<Tuple> {

    private ColumnRawStatsService columnRawStatsService;
    private ColumnBinningService columnBinningService;
    private ColumnNumStatsService columnNumStatsService;
    private ColumnBinStatsService columnBinStatsService;

    private Double valueThreshold = 1e6;

    private ObjectMapper jsonMapper;

    public CalculateStatsUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException  {
        super(source, pathModelConfig, pathColumnConfig);

        if ( modelConfig.getNumericalValueThreshold() != null ) {
            valueThreshold = modelConfig.getNumericalValueThreshold();
        }

        StatsModule statsModule = new StatsModule();
        statsModule.setRawStatsCalculatorImplClass(modelConfig.getStats().getRawStatsCalculator());
        statsModule.setNumBinningCalculatorImplClass(modelConfig.getStats().getNumBinningCalculator());
        statsModule.setCatBinningCalculatorImplClass(modelConfig.getStats().getCatBinningCalculator());
        statsModule.setNumStatsCalculatorImplClass(modelConfig.getStats().getNumStatsCalculator());
        statsModule.setBinStatsCalculatorImplClass(modelConfig.getStats().getBinStatsCalculator());

        Injector injector = Guice.createInjector(statsModule);

        columnRawStatsService = injector.getInstance(ColumnRawStatsService.class);
        columnBinningService = injector.getInstance(ColumnBinningService.class);
        columnNumStatsService = injector.getInstance(ColumnNumStatsService.class);
        columnBinStatsService = injector.getInstance(ColumnBinStatsService.class);

        jsonMapper = new ObjectMapper();

        //log.debug("Value Threshold: " + valueThreshold);
    }

    public Tuple exec(Tuple input) throws IOException {
        if (input == null || input.size() == 0) {
            return null;
        }

        TupleFactory tupleFactory = TupleFactory.getInstance();

        Integer columnNum = (Integer) input.get(0);
        DataBag bag = (DataBag) input.get(1);

        ColumnConfig columnConfig = columnConfigList.get(columnNum);

        List<RawValueObject> rvoList = new ArrayList<RawValueObject>();

        log.debug("****** The element count in bag is : " + bag.size());



        for (Tuple t : bag) {
            RawValueObject rvo = new RawValueObject();
            rvo.setValue(t.get(0));
            rvo.setTag(t.get(1).toString());
            rvo.setWeight(Double.valueOf(t.get(2).toString()));
            rvoList.add(rvo);
        }

        ColumnRawStatsResult screeningResult = columnRawStatsService.getResult(rvoList, modelConfig.getPosTags(), modelConfig.getNegTags());
        columnConfig.setColumnRawStatsResult(screeningResult);

        //TODO: Let user choose if column should be treated as Numerical or Categorical if not predefined in ColumnConfig

        ColumnBinningResult binningResult;
        ColumnNumStatsResult basicStats;

        if (columnConfig.isNumerical()) {
            List<NumericalValueObject> nvoList = CommonUtils.convertListRaw2Numerical(rvoList, modelConfig.getPosTags(), modelConfig.getNegTags());
            binningResult = columnBinningService.getNumericalResult(nvoList, modelConfig.getBinningExpectedNum());
            basicStats = columnNumStatsService.getResult(nvoList);
        } else {
            List<CategoricalValueObject> cvoList = CommonUtils.convertListRaw2Categorical(rvoList, modelConfig.getPosTags(), modelConfig.getNegTags());
            binningResult = columnBinningService.getCategoricalResult(cvoList);
            basicStats = columnNumStatsService.getResult(CommonUtils.convertListCategorical2Numerical(cvoList, binningResult));
        }
        columnConfig.setColumnBinningResult(binningResult);
        columnConfig.setColumnNumStatsResult(basicStats);


        // Advanced Stats

        ColumnBinStatsResult advStats = columnBinStatsService.getResult(binningResult);

        columnConfig.setColumnBinStatsResult(advStats);

        Tuple tuple = tupleFactory.newTuple();
        tuple.append(columnNum);
        tuple.append(jsonMapper.writeValueAsString(columnConfig));

        return tuple;

    }

}
