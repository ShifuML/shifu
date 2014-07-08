package ml.shifu.plugin.encog.adapter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import ml.shifu.core.plugin.pmml.AdapterConstants;
import ml.shifu.core.plugin.pmml.PMMLAdapterCommonUtil;
import ml.shifu.core.plugin.pmml.PMMLModelBuilder;

import org.dmg.pmml.ActivationFunctionType;
import org.dmg.pmml.Connection;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.LocalTransformations;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.NeuralInput;
import org.dmg.pmml.NeuralInputs;
import org.dmg.pmml.NeuralLayer;
import org.dmg.pmml.Neuron;
import org.dmg.pmml.OpType;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.neural.flat.FlatNetwork;

/**
 * The class that converts an Encog NeuralNetwork to a PMML RegressionModel.
 * This class extends the abstract class
 * PMMLModelBuilder<pmml.RegressionModel,Encog.NeuralNetwork>.
 * 
 */
public class PMMLEncogNeuralNetworkModel
        implements
        PMMLModelBuilder<org.dmg.pmml.NeuralNetwork, org.encog.neural.networks.BasicNetwork> {

    private FlatNetwork network;

    /**
     * <p>
     * The function which converts an Encog NeuralNetwork to a PMML
     * NeuralNetwork Model.
     * <p>
     * This function reads the weights from the Encog NeuralNetwork model and
     * assign them to the corresponding connections of Neurons in PMML model.
     * 
     * @param bNetwork
     *            Encog NeuralNetwork
     * @param utility
     *            DataFieldUtility that provides supplementary data field for
     *            the model conversion
     * @return The generated PMML NeuralNetwork Model
     */
    public org.dmg.pmml.NeuralNetwork adaptMLModelToPMML(
            org.encog.neural.networks.BasicNetwork bNetwork,
            org.dmg.pmml.NeuralNetwork pmmlModel) {
        network = bNetwork.getFlat();
        MiningSchema schema = pmmlModel.getMiningSchema();
//        pmmlModel.withNeuralInputs(PMMLAdapterCommonUtil
//                .getNeuralInputs(schema));
//        pmmlModel
//                .withLocalTransformations(PMMLAdapterCommonUtil
//                        .getBiasLocalTransformation(pmmlModel
//                                .getLocalTransformations()));
//         deleteTargetDerivedFields(pmmlModel);
        renameDerivedFields(pmmlModel);
        int[] layerCount = network.getLayerCounts();
        int[] layerFeedCount = network.getLayerFeedCounts();
        double[] weights = network.getWeights();
        ActivationFunctionType[] functionList = transformActivationFunction(network
                .getActivationFunctions());

        int numLayers = layerCount.length;
        int weightID = 0;
        List<NeuralLayer> layerList = new ArrayList<NeuralLayer>();
        String biasValue = "bias";
        pmmlModel.withFunctionName(MiningFunctionType.REGRESSION);
        for (int i = 0; i < numLayers - 1; i++) {
            NeuralLayer layer = new NeuralLayer();
            layer.setNumberOfNeurons(layerFeedCount[i]);
            layer.setActivationFunction(functionList[i]);
            int layerID = numLayers - i - 1;
            for (int j = 0; j < layerFeedCount[i]; j++) {
                Neuron neuron = new Neuron(String.valueOf(layerID + "," + j));
                neuron.setBias(0.0);// bias of each neuron, set to 0

                for (int k = 0; k < layerFeedCount[i + 1]; k++) {
                    neuron.withConnections(new Connection(String
                            .valueOf(layerID - 1 + "," + k),
                            weights[weightID++]));
                }// weights
                int tmp = layerCount[i + 1] - layerFeedCount[i + 1];
                // TODO set bias as constant, don't need to read from field
                for (int k = 0; k < tmp; k++) {
                    neuron.withConnections(new Connection(biasValue,
                            weights[weightID++]));
                }// bias neuron for each layer, set to bias=1
                layer.withNeurons(neuron);
            }// finish build Neuron
            layerList.add(layer);
        }// finish build layer
         // reserve the layer list to fit fot PMML format
        Collections.reverse(layerList);
        pmmlModel.withNeuralLayers(layerList);
        // set neural output based on target id
        pmmlModel.withNeuralOutputs(PMMLAdapterCommonUtil.getOutputFields(
                schema, numLayers - 1));
        // deleteTargetDerivedFields(pmmlModel);
        return pmmlModel;
    }

    private ActivationFunctionType[] transformActivationFunction(
            ActivationFunction[] functions) {
        int funLen = functions.length;
        ActivationFunctionType[] functionType = new ActivationFunctionType[funLen];
        @SuppressWarnings("serial")
        HashMap<String, ActivationFunctionType> functionMap = new HashMap<String, ActivationFunctionType>() {
            {
                put("ActivationSigmoid", ActivationFunctionType.LOGISTIC);
                put("ActivationLinear", ActivationFunctionType.IDENTITY);
                put("ActivationTANH", ActivationFunctionType.TANH);
            }
        };
        for (int i = 0; i < funLen; i++) {
            String trimS = functions[i].getClass().getName();
            String[] functionS = trimS.split("\\.");
            functionType[i] = functionMap.get(functionS[functionS.length - 1]);
        }
        return functionType;
    }

    private org.dmg.pmml.NeuralNetwork renameDerivedFields(
            org.dmg.pmml.NeuralNetwork pmmlModel) {
        // delete target
        List<DerivedField> derivedFields = pmmlModel.getLocalTransformations()
                .getDerivedFields();
        derivedFields.remove(0);
        // change name
        for (DerivedField field : derivedFields) {
            String name = field.getName().getValue();
            field.setName(new FieldName(name + "_T"));
        }
        // add bias
        DerivedField field = new DerivedField(OpType.CONTINUOUS,
                DataType.DOUBLE).withName(new FieldName(
                AdapterConstants.biasValue));
        // field.withName(new FieldName(s));
        field.withExpression(new Constant(String.valueOf(AdapterConstants.bias)));
        derivedFields.add(field);
        pmmlModel.setLocalTransformations(new LocalTransformations()
                .withDerivedFields(derivedFields));
        int index = 0;
        NeuralInputs inputs = new NeuralInputs();
        // add input
        for (int i = 0; i < derivedFields.size() - 1; i++) {
            String name = derivedFields.get(i).getName().getValue();
            DerivedField inputF = new DerivedField(OpType.CONTINUOUS,
                    DataType.DOUBLE).withName(new FieldName(name))
                    .withExpression(new FieldRef(new FieldName(name)));
            inputs.withNeuralInputs(new NeuralInput(inputF, "0," + (index++)));
        }
        DerivedField biasF = new DerivedField(OpType.CONTINUOUS,
                DataType.DOUBLE).withName(
                new FieldName(AdapterConstants.biasValue)).withExpression(
                new FieldRef(new FieldName(AdapterConstants.biasValue)));
        inputs.withNeuralInputs(new NeuralInput(biasF,
                AdapterConstants.biasValue));

        pmmlModel.setNeuralInputs(inputs);

        return pmmlModel;
    }

    @SuppressWarnings("unused")
    private org.dmg.pmml.NeuralNetwork deleteTargetDerivedFields(
            org.dmg.pmml.NeuralNetwork pmmlModel) {
        // delete target
       pmmlModel.getLocalTransformations()
                .getDerivedFields().remove(0);
        return pmmlModel;
    }
}
