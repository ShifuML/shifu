/**
 *
 */
package ml.shifu.core.plugin.pmml;

import org.dmg.pmml.Model;

/**
 * The abstract class that converts the Machine Learning model to a PMML model
 *
 * @param <T> The target PMML model type
 * @param <S> The source ML model from specific Machine Learning framework such
 *            as Encog, Mahout, and Spark.
 */
public interface PMMLModelBuilder<T extends Model, S> {


    T adaptMLModelToPMML(S mlModel, T partialPMMLModel);

}
