/*
 * Copyright [2012-2014] PayPal Software Foundation
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

import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.Reasoner;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * CalculateReasonCodeUDF class is to calculate the reason code for each evaluation data
 */
public class CalculateReasonCodeUDF extends AbstractTrainerUDF<Tuple> {

    private Map<String, String> reasonCodeMap;
    private String[] headers;

    public CalculateReasonCodeUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName)
            throws Exception {
        super(source, pathModelConfig, pathColumnConfig);

        EvalConfig evalConfig = modelConfig.getEvalConfigByName(evalSetName);

        SourceType sourceType = evalConfig.getDataSet().getSource();
        // CommonUtils.determineSource(modelConfig.getRunConf().getRunMode(),
        // evalConfig.getDataSet().getSource());
        PathFinder pathFinder = new PathFinder(modelConfig);
        reasonCodeMap = CommonUtils
                .loadAndFlattenReasonCodeMap(pathFinder.getReasonCodeMapPath(sourceType), sourceType);

        headers = CommonUtils.getFinalHeaders(evalConfig);

        log.debug("The length of header is: " + headers.length);
    }

    public Tuple exec(Tuple input) throws IOException {
        Tuple result = TupleFactory.getInstance().newTuple();

        if(input == null || input.size() == 0 || headers.length == 0 || input.size() != headers.length) {
            return null;
        }

        Map<String, String> rawDataMap = new HashMap<String, String>();

        try {
            for(int i = 0; i < headers.length; i++) {
                Object t = input.get(i);
                if(t == null) {
                    continue;
                }
                rawDataMap.put(headers[i], t.toString());
                // log.info(headers[i] + ", " + t.toString());
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        // result.append(rawDataMap.toString());
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        Reasoner reasoner = new Reasoner(this.reasonCodeMap);
        result.append(rawDataMap.get("UnauthScore"));
        result.append(rawDataMap.get("TxnSnrId"));
        result.append(rawDataMap.get("TxnCpId"));
        result.append(sdf.format(new Date(1000 * Long.parseLong(rawDataMap.get("TxnCreatedTs")))));
        result.append(rawDataMap.get("TxnAmtUsd"));
        result.append(rawDataMap.get("ActivityID"));
        reasoner.calculateReasonCodes(columnConfigList, rawDataMap);

        List<String> reasons = reasoner.getReasonCodes();

        try {
            if(reasons != null) {
                for(String reason: reasons) {
                    result.append(reason);
                }
            } else {
                result.append("Error: No Reason Code Returned.");
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
        return result;
    }

    public Schema outputSchema(Schema input) {
        // Utils.getSchemaFromString(schemaString)
        return null;
    }

}
