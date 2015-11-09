/**
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

package ml.shifu.shifu.core.pmml;

import java.util.List;

import org.dmg.pmml.DerivedField;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.NumericPredictor;
import org.dmg.pmml.RegressionNormalizationMethodType;
import org.dmg.pmml.RegressionTable;

/**
 * The class that converts an LR model to a PMML RegressionModel.
 * This class extends the abstract class
 * PMMLModelBuilder<pmml.RegressionModel,LR>.
 */
public class PMMLLRModelBuilder
        implements
        PMMLModelBuilder<org.dmg.pmml.RegressionModel, ml.shifu.shifu.core.LR> {

    public org.dmg.pmml.RegressionModel adaptMLModelToPMML(
            ml.shifu.shifu.core.LR lr,
            org.dmg.pmml.RegressionModel pmmlModel) {
        pmmlModel.setNormalizationMethod(RegressionNormalizationMethodType.LOGIT);
        pmmlModel.setFunctionName(MiningFunctionType.REGRESSION);
        RegressionTable table = new RegressionTable();
        table.setIntercept(lr.getBias());
        LocalTransformations lt = pmmlModel.getLocalTransformations();
        List<DerivedField> df = lt.getDerivedFields();
        for(int i = 0; i<df.size();i++){
            DerivedField field = df.get(i);
            NumericPredictor np = new NumericPredictor();
            np.setName(field.getName());
            np.setCoefficient(lr.getWeights()[i]);
            table.withNumericPredictors(np);
        }
        pmmlModel.withRegressionTables(table);
        return pmmlModel;
    }
}
