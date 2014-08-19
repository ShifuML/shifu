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
package ml.shifu.plugin.pig.stats;

import ml.shifu.core.request.Request;
import ml.shifu.core.util.Params;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.pig.EvalFunc;
import org.apache.pig.data.BagFactory;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.dmg.pmml.DataField;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.mortbay.log.Log;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.xml.sax.InputSource;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import javax.xml.transform.sax.SAXSource;

/**
 * <pre>
 * AddColumnNumUDF class is to convert tuple of row data into bag of column data
 * Its structure is like
 *    {
 * 		(column-id, column-value, column-tag, column-score)
 * 		(column-id, column-value, column-tag, column-score)
 * 		...
 * }
 */
public class PigAddColumnNumUDF extends EvalFunc<DataBag> {
    private static Logger log = LoggerFactory
            .getLogger(PigStatsRequestProcessor.class);

    private List<String> negTags;
    // private List<String> posTags;

    private PMML pmml;

    private Random random = new Random(System.currentTimeMillis());

    private int weightedColumnNum = -1;

    private String modelName;

    private double sampleRate;

    private boolean isCategoricalDisabled;

    private boolean binningSampleNegOnly;

    public PigAddColumnNumUDF(String request) throws Exception {

        ObjectMapper jsonMapper = new ObjectMapper();
        Request req = jsonMapper.readValue(request, Request.class);
        Params params = req.getProcessor().getParams();
        Params bindings = req.getBindings().get(0).getParams();

        this.pmml = loadPMML((String) params.get("pathPMML"));
        this.modelName = (String) bindings.get("modelName");
        this.isCategoricalDisabled = (Boolean) params
                .get("isCategoricalDisabled");
        this.binningSampleNegOnly = (Boolean) params
                .get("binningSampleNegOnly");

        try {
            this.sampleRate = Double.parseDouble((String) params
                    .get("sampleRate"));
        } catch (Exception e) {
            this.sampleRate = 1.0;
        }

        negTags = (List<String>) bindings.get("negTags");

    }

    private Params parseParams(Params rawParams) throws Exception {
        ObjectMapper jsonMapper = new ObjectMapper();
        String jsonString = jsonMapper.writeValueAsString(rawParams);
        return jsonMapper.readValue(jsonString, Params.class);
    }

    private static PMML loadPMML(String path) throws Exception {

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path pmmlFilePath = new Path(path);
        FSDataInputStream in = fs.open(pmmlFilePath);

        try {
            InputSource source = new InputSource(in);
            SAXSource transformedSource = ImportFilter.apply(source);
            return JAXBUtil.unmarshalPMML(transformedSource);

        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }
    }

    public DataBag exec(Tuple input) throws IOException {

        DataBag bag = BagFactory.getInstance().newDefaultBag();
        TupleFactory tupleFactory = TupleFactory.getInstance();

        if (input == null) {
            return null;
        }

        int dataSize = pmml.getDataDictionary().getDataFields().size();
        if (dataSize != input.size()) {
            throw new RuntimeException("Data Mismatch: dictionary fields = "
                    + dataSize + ", data fields: " + input.size());
        }

        Model model = null;
        for (Model m : pmml.getModels()) {
            if (m.getModelName().equals(this.modelName)) {
                model = m;
            }
        }
        if (model == null)
            return null;

        String target = "";

        if (binningSampleNegOnly) {
            for (MiningField miningField : model.getMiningSchema()
                    .getMiningFields()) {
                if (miningField.getUsageType().equals(FieldUsageType.TARGET)) {
                    target = miningField.getName().getValue();
                    break;
                }
            }
            if (negTags.contains(target) && random.nextDouble() > sampleRate) {
                return null;
            }
        } else {
            if (random.nextDouble() > sampleRate) {
                return null;
            }
        }

        int varSize = input.size();

        Map<String, Object> usageTypeMap = new HashMap<String, Object>();
        for (MiningField miningField : model.getMiningSchema()
                .getMiningFields()) {
            usageTypeMap.put(miningField.getName().toString(),
                    miningField.getUsageType());
        }

        Map<String, Object> rawDataMap = new HashMap<String, Object>();
        for (int i = 0; i < input.size(); i++) {
            rawDataMap.put(pmml.getDataDictionary().getDataFields().get(i)
                    .getName().toString(), input.get(i).toString());
        }

        for (int i = 0; i < varSize; i++) {

            if (isCategoricalDisabled) {
                try {
                    Double.valueOf(input.get(i).toString());
                } catch (Exception e) {
                    continue;
                }
            }
            String fieldName = pmml.getDataDictionary().getDataFields().get(i)
                    .getName().toString();

            if (usageTypeMap.get(fieldName) == FieldUsageType.ACTIVE) {
                Tuple tuple = tupleFactory.newTuple(4);
                tuple.set(0, i);

                // Set Data
                tuple.set(1, input.get(i));

                // Set Tag
                tuple.set(2, rawDataMap.get(target));

                // set weights
                if (weightedColumnNum != -1) {
                    try {
                        tuple.set(3, Double.valueOf(input
                                .get(weightedColumnNum).toString()));
                    } catch (NumberFormatException e) {
                        tuple.set(3, 1.0);
                    }

                    if (i == weightedColumnNum) {
                        // weight and its column, set to 1
                        tuple.set(3, 1.0);
                    }
                } else {
                    tuple.set(3, 1.0);
                }

                bag.add(tuple);
            }
        }

        return bag;
    }

    public Schema outputSchema(Schema input) {
        return null;
    }
}