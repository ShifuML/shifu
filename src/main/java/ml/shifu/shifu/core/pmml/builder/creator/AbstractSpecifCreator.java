package ml.shifu.shifu.core.pmml.builder.creator;

import org.dmg.pmml.*;
import org.encog.ml.BasicML;

/**
 * Created by zhanhu on 3/30/16.
 */
public abstract class AbstractSpecifCreator {

    public static final String RAW_RESULT = "RawResult";
    public static final String ROUND_FUNC = "round";
    public static final String FINAL_RESULT = "FinalResult";

    public abstract boolean build(BasicML basicML, Model model);

    /**
     * Create the normalized output for model, since the final score should be 0 ~ 1000, instead of 0.o ~ 1.0
     *
     * @return @Output for model
     */
    protected Output createNormalizedOutput() {
        Output output = new Output();

        output.withOutputFields(createOutputField(RAW_RESULT, OpType.CONTINUOUS, DataType.DOUBLE,
                ResultFeatureType.PREDICTED_VALUE));

        OutputField finalResult = createOutputField(FINAL_RESULT, OpType.CONTINUOUS, DataType.DOUBLE,
                ResultFeatureType.TRANSFORMED_VALUE);
        finalResult.withExpression(createApplyFunc());

        output.withOutputFields(finalResult);

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
     * @param feature
     *            - result feature type
     * @return @OutputField
     */
    protected OutputField createOutputField(String fieldName, OpType opType, DataType dataType, ResultFeatureType feature) {
        OutputField outputField = new OutputField();
        outputField.withName(new FieldName(fieldName));
        outputField.withOptype(opType);
        outputField.withDataType(dataType);
        outputField.withFeature(feature);
        return outputField;
    }

    /**
     * Create the apply expression for final output, the function is "round"
     *
     * @return @Apply
     */
    protected Apply createApplyFunc() {
        Apply apply = new Apply();

        apply.withFunction(ROUND_FUNC);

        NormContinuous normContinuous = new NormContinuous();
        normContinuous.withField(new FieldName(RAW_RESULT));
        normContinuous.withLinearNorms(new LinearNorm().withOrig(0).withNorm(0));
        normContinuous.withLinearNorms(new LinearNorm().withOrig(1).withNorm(1000));

        apply.withExpressions(normContinuous);

        return apply;
    }

}
