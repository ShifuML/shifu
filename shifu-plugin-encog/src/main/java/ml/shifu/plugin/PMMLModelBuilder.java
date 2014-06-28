/**
 * 
 */
package ml.shifu.plugin;

import org.dmg.pmml.Model;

/**
 * The abstract class that converts the Machine Learing model to a PMML model
 * 
 * @param <T>
 *            The target PMML model type
 * @param <S>
 *            The source ML model from specific Machine Learning framework such
 *            as Encog, Machout, and Spark.
 */
public interface PMMLModelBuilder<T extends Model,S> {

    /**
     * The function which converts the Machine Learning model to a PMML model.
     * 
     * @param model
     *            The model from ML frameworks
     * @param utility
     *            DataFieldUtility that provides supplementary data field for
     *            the model conversion
     * @return The PMMLModel after conversion
     */
   T adaptMLModelToPMML(S mlModel, T partialPMMLModel);

}
