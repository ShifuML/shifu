/*
 * Copyright [2013-2016] PayPal Software Foundation
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
package ml.shifu.shifu.core.pmml.builder.creator;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import org.dmg.pmml.*;
import org.encog.ml.BasicML;

import java.util.List;

/**
 * Created by zhanhu on 3/30/16.
 */
public abstract class AbstractSpecifCreator {

    public static final String RAW_RESULT = "RawResult";
    public static final String ROUND_FUNC = "round";
    public static final String FINAL_RESULT = "FinalResult";

    private ModelConfig modelConfig;
    @SuppressWarnings("unused")
    private List<ColumnConfig> columnConfigList;

    public AbstractSpecifCreator(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
    }

    public abstract boolean build(BasicML basicML, Model model);

    public abstract boolean build(BasicML basicML, Model model, int id);

    /**
     * Create the normalized output for model, since the final score should be 0 ~ 1000, instead of 0.o ~ 1.0
     * 
     * @return output for model
     */
    protected Output createNormalizedOutput() {
        Output output = new Output();
        if ( modelConfig.isClassification() &&
                ModelTrainConf.MultipleClassification.NATIVE.equals(modelConfig.getTrain().getMultiClassifyMethod()) ) {
            for ( int i = 0; i < modelConfig.getTags().size(); i ++ ) {
                output.addOutputFields(createOutputField(RAW_RESULT + "_" + i, OpType.CONTINUOUS, DataType.DOUBLE,
                        new FieldName(modelConfig.getTargetColumnName() + "_" + i), ResultFeature.PREDICTED_VALUE));

                OutputField finalResult = createOutputField(FINAL_RESULT + "_" + i, OpType.CONTINUOUS, DataType.DOUBLE,
                        new FieldName(modelConfig.getTargetColumnName() + "_" + i), ResultFeature.TRANSFORMED_VALUE);
                finalResult.setExpression(createNormExpr(i));

                output.addOutputFields(finalResult);
            }
        } else {
            output.addOutputFields(createOutputField(RAW_RESULT, OpType.CONTINUOUS, DataType.DOUBLE,
                    new FieldName(modelConfig.getTargetColumnName()), ResultFeature.PREDICTED_VALUE));

            OutputField finalResult = createOutputField(FINAL_RESULT, OpType.CONTINUOUS, DataType.DOUBLE,
                    new FieldName(modelConfig.getTargetColumnName()), ResultFeature.TRANSFORMED_VALUE);
            finalResult.setExpression(createNormExpr());

            output.addOutputFields(finalResult);
        }
        return output;
    }

    /**
     * Create the normalized output for model, since the final score should be 0 ~ 1000, instead of 0.o ~ 1.0
     * 
     * @param id
     *            output id
     * @return output for model
     */
    protected Output createNormalizedOutput(int id) {
        Output output = new Output();

        output.addOutputFields(createOutputField(RAW_RESULT + "_" + id, OpType.CONTINUOUS, DataType.DOUBLE,
                new FieldName(modelConfig.getTargetColumnName() + "_" + id), ResultFeature.PREDICTED_VALUE));

        OutputField finalResult = createOutputField(FINAL_RESULT + "_" + id, OpType.CONTINUOUS, DataType.DOUBLE,
                new FieldName(modelConfig.getTargetColumnName() + "_" + id), ResultFeature.TRANSFORMED_VALUE);
        finalResult.setExpression(createNormExpr(id));

        output.addOutputFields(finalResult);

        return output;
    }

    /**
     * Create the output field, and set the field name, operation type, data type and feature type
     * 
     * @param fieldName
     *            - the name of output field
     * @param opType
     *            - operation type
     * @param dataType
     *            - data type
     * @param targetField
     *            - the target field name
     * @param feature
     *            - result feature type
     * @return OutputField
     */
    protected OutputField createOutputField(String fieldName, OpType opType, DataType dataType,
            FieldName targetField, ResultFeature feature) {
        OutputField outputField = new OutputField();
        outputField.setName(new FieldName(fieldName));
        outputField.setOpType(opType);
        outputField.setDataType(dataType);
        outputField.setResultFeature(feature);
        if ( targetField != null ) {
            outputField.setTargetField(targetField);
        }
        return outputField;
    }

    /**
     * Create the apply expression for final output, the function is "round"
     * 
     * @return Apply
     */
    protected Expression createNormExpr() {
        NormContinuous normContinuous = new NormContinuous();
        normContinuous.setField(new FieldName(RAW_RESULT));
        normContinuous.addLinearNorms(new LinearNorm().setOrig(0).setNorm(0));
        normContinuous.addLinearNorms(new LinearNorm().setOrig(1).setNorm(1000));
        return normContinuous;
    }

    protected Expression createNormExpr(int id) {
        NormContinuous normContinuous = new NormContinuous();
        normContinuous.setField(new FieldName(RAW_RESULT + "_" + id));
        normContinuous.addLinearNorms(new LinearNorm().setOrig(0).setNorm(0));
        normContinuous.addLinearNorms(new LinearNorm().setOrig(1).setNorm(1000));
        return normContinuous;
    }
}
