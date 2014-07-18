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

package ml.shifu.plugin.mahout.adapter;

import java.util.List;

import ml.shifu.core.plugin.pmml.AdapterConstants;
import ml.shifu.core.plugin.pmml.PMMLAdapterCommonUtil;
import ml.shifu.core.plugin.pmml.PMMLModelBuilder;

import org.apache.mahout.classifier.mlp.NeuralNetwork;
import org.apache.mahout.math.Matrix;
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

public class PMMLMahoutNeuralNetworkModel implements
		PMMLModelBuilder<org.dmg.pmml.NeuralNetwork, NeuralNetwork> {

	public org.dmg.pmml.NeuralNetwork adaptMLModelToPMML(NeuralNetwork nnModel,
			org.dmg.pmml.NeuralNetwork pmmlModel) {
		Matrix[] matrixList = nnModel.getWeightMatrices();
		MiningSchema schema = pmmlModel.getMiningSchema();
		pmmlModel.withFunctionName(MiningFunctionType.REGRESSION);
		renameDerivedFields(pmmlModel);
		for (int layerIndex = 0; layerIndex < matrixList.length; layerIndex++) {
			NeuralLayer layer = new NeuralLayer();
			Matrix matrix = matrixList[layerIndex];
			int rowSize = matrix.numRows();
			int columnSize = matrix.numCols();
			layer.setNumberOfNeurons(rowSize);
			// TODO since squashFunctionList in Mahout is not accessible, by
			// default, we set ActivationFunction to sigmoid
			layer.setActivationFunction(ActivationFunctionType.LOGISTIC);
			for (int j = 0; j < rowSize; j++) {
				Neuron neuron = new Neuron(String.valueOf((layerIndex + 1)
						+ "," + j));
				neuron.setBias(0.0);// bias of each neuron, set to 0

				for (int k = 1; k < columnSize; k++) {
					neuron.withConnections(new Connection(String
							.valueOf(layerIndex + "," + (k - 1)), matrix.get(j,
							k)));
				}
				// bias neuron for each layer, set to bias=1
				neuron.withConnections(new Connection(
						AdapterConstants.biasValue, matrix.get(j, 0)));
				layer.withNeurons(neuron);
			}// finish build Neuron
			pmmlModel.withNeuralLayers(layer);
		}// finish build layer
			// TODO outputID: classify to M or B based on the input
		pmmlModel.withNeuralOutputs(PMMLAdapterCommonUtil.getOutputFields(
				schema, matrixList.length));
		return pmmlModel;
	}

	private org.dmg.pmml.NeuralNetwork renameDerivedFields(
			org.dmg.pmml.NeuralNetwork pmmlModel) {
		// delete target
		List<DerivedField> derivedFields = pmmlModel.getLocalTransformations()
				.getDerivedFields();
		derivedFields.remove(0);
		// change name
//		for (DerivedField field : derivedFields) {
//			String name = field.getName().getValue();
//			field.setName(new FieldName(name + "_T"));
//		}
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
		pmmlModel.getLocalTransformations().getDerivedFields().remove(0);
		return pmmlModel;
	}
}
