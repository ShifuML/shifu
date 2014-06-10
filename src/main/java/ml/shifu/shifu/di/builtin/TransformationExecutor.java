package ml.shifu.shifu.di.builtin;

import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.NormContinuous;
import org.jpmml.evaluator.NormalizationUtil;

public class TransformationExecutor {

    public static Object transform(DerivedField derivedField, Object origin) {

        Expression expression = derivedField.getExpression();

        if (expression instanceof NormContinuous) {
            return NormalizationUtil.normalize((NormContinuous) expression, Double.parseDouble(origin.toString()));
        } else {
            throw new RuntimeException("Invalid Expression");
        }

    }


}
