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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.container.CategoricalValueObject;
import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.container.obj.*;
import ml.shifu.shifu.di.module.StatsModule;
import ml.shifu.shifu.di.service.*;
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

    private StatsService statsService;

    private Double valueThreshold = 1e6;

    private ObjectMapper jsonMapper;

    public CalculateStatsUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException  {
        super(source, pathModelConfig, pathColumnConfig);

        if ( modelConfig.getNumericalValueThreshold() != null ) {
            valueThreshold = modelConfig.getNumericalValueThreshold();
        }

        StatsModule statsModule = new StatsModule();
        statsModule.setMethods(modelConfig.getStats().getMethods());

        //statsModule.setStatsProcessorImplClass(modelConfig.getStats().getStatsProcessor());
       // statsModule.setRawStatsCalculatorImplClass(modelConfig.getStats().getRawStatsCalculator());
       // statsModule.setNumBinningCalculatorImplClass(modelConfig.getStats().getNumBinningCalculator());
       // statsModule.setCatBinningCalculatorImplClass(modelConfig.getStats().getCatBinningCalculator());
       // statsModule.setNumStatsCalculatorImplClass(modelConfig.getStats().getNumStatsCalculator());
       // statsModule.setBinStatsCalculatorImplClass(modelConfig.getStats().getBinStatsCalculator());

        Injector injector = Guice.createInjector(statsModule);

        statsService = injector.getInstance(StatsService.class);

        Map<String, Object> params = new HashMap<String, Object>();

        params.put("numBins", modelConfig.getBinningExpectedNum());
        params.put("posTags", modelConfig.getPosTags());
        params.put("negTags", modelConfig.getNegTags());
        statsService.setParams(params);

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

        statsService.exec(columnConfig, rvoList);

        Tuple tuple = tupleFactory.newTuple();
        tuple.append(columnNum);
        tuple.append(jsonMapper.writeValueAsString(columnConfig));

        return tuple;

    }

}
