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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Calculate the score for each evaluation data
 */
public class EncodeDataUDF extends AbstractEvalUDF<Tuple> {

    private PathFinder pathFinder;
    private IndependentTreeModel treeModel;

    public EncodeDataUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig, evalSetName);
        this.pathFinder = new PathFinder(this.modelConfig);

        // get model path
        SourceType sourceType = SourceType.valueOf(source);
        FileSystem fileSystem = ShifuFileUtils.getFileSystemBySourceType(sourceType);
        Path modelPath = fileSystem.makeQualified(new Path(this.pathFinder.getModelsPath(sourceType), getModelName(0)));

        // load Tree model
        InputStream inputStream = null;
        try {
            inputStream = ShifuFileUtils.getInputStream(modelPath, sourceType);
            this.treeModel = IndependentTreeModel.loadFromStream(inputStream);
        } finally {
            IOUtils.closeQuietly(inputStream);
        }

        if(StringUtils.isNotBlank(evalSetName)) {
            for(EvalConfig evalConfig : this.modelConfig.getEvals()) {
                if(evalConfig.getName().equals(evalSetName)) {
                    this.evalConfig = evalConfig;
                    break;
                }
            }
        }
    }

    public String getModelName(int i) {
        String alg = this.modelConfig.getTrain().getAlgorithm();
        return String.format("model%s.%s", i, alg.toLowerCase());
    }

    @Override public Tuple exec(Tuple tuple) throws IOException {
        List<String> outputList = new ArrayList<String>();

        Map<String, Object> rawInput = convertTupleToRawInput(tuple);
        //1. append the tag
        Object obj = rawInput.get(this.modelConfig.getTargetColumnName(this.evalConfig,
                this.modelConfig.getTargetColumnName()));
        outputList.add(obj == null ? null : obj.toString());

        //2. append the weight
        String weightColumn = ((evalConfig == null) ? this.modelConfig.getWeightColumnName()
                : this.evalConfig.getDataSet().getWeightColumnName());
        if (StringUtils.isNotBlank(weightColumn)) {
            obj = rawInput.get(weightColumn);
            outputList.add(obj == null ? null : obj.toString());
        } else {
            outputList.add("1.0");
        }

        int depth = Integer.parseInt(this.modelConfig.getParams().get("MaxDepth").toString());
        outputList.addAll(treeModel.encode(depth - 1, rawInput));

        if ( evalConfig == null ) {
            for(ColumnConfig columnConfig : this.columnConfigList) {
                if(ColumnConfig.ColumnFlag.Meta.equals(columnConfig.getColumnFlag())) { // only Meta, skip Weight
                    obj = rawInput.get(columnConfig.getColumnName());
                    outputList.add(obj == null ? null : obj.toString());
                }
            }
        } else {
            for (String metaColumn : this.evalConfig.getAllMetaColumns(this.modelConfig)) {
                obj = rawInput.get(metaColumn);
                outputList.add(obj == null ? null : obj.toString());
            }
        }

        return TupleFactory.getInstance().newTuple(outputList);
    }

    private Map<String, Object> convertTupleToRawInput(Tuple tuple) throws ExecException {
        Map<String, Object> rawInput = new HashMap<String, Object>();
        for(int i = 0; i < this.columnConfigList.size(); i++) {
            rawInput.put(this.columnConfigList.get(i).getColumnName(), tuple.get(i));
        }
        return rawInput;
    }

    /**
     * output the schema for evaluation score
     */
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            String targetColumnName = this.modelConfig.getTargetColumnName(this.evalConfig,
                    modelConfig.getTargetColumnName());
            tupleSchema.add(new FieldSchema(targetColumnName, DataType.CHARARRAY));

            String weightName = this.modelConfig.getWeightColumnName();
            if(evalConfig != null) {
                weightName = evalConfig.getDataSet().getWeightColumnName();
            }
            weightName = (StringUtils.isBlank(weightName) ? "weight" : weightName);
            tupleSchema.add(new FieldSchema(weightName, DataType.CHARARRAY));

            int featureCnt = 1;
            for(int i = 0; i < this.treeModel.getTrees().size(); i++) {
                featureCnt = featureCnt * this.treeModel.getTrees().get(i).size();
            }

            for(int i = 0; i < featureCnt; i++) {
                tupleSchema.add(new FieldSchema("tree_vars_" + i, DataType.CHARARRAY));
            }

            if(evalConfig == null) {
                for(ColumnConfig columnConfig : this.columnConfigList) {
                    if(ColumnConfig.ColumnFlag.Meta.equals(columnConfig.getColumnFlag())) { // only Meta, skip Weight
                        tupleSchema.add(new FieldSchema(columnConfig.getColumnName(), DataType.CHARARRAY));
                    }
                }
            } else {
                for(String metaColumn : evalConfig.getAllMetaColumns(this.modelConfig)) {
                    tupleSchema.add(new FieldSchema(metaColumn, DataType.CHARARRAY));
                }
            }

            return new Schema(new FieldSchema("EncodeData", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }
}
