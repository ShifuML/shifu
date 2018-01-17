/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.dataset;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ml.shifu.shifu.core.dtrain.nn.ActivationReLU;

import org.apache.commons.lang.StringUtils;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.PersistBasicNetwork;
import org.encog.persist.EncogFileSection;
import org.encog.persist.EncogPersistor;
import org.encog.persist.EncogReadHelper;
import org.encog.persist.EncogWriteHelper;
import org.encog.persist.PersistConst;
import org.encog.persist.PersistError;
import org.encog.util.csv.CSVFormat;

/**
 * Support {@link BasicFloatNetwork} serialization and de-serialization. This is copied from {@link PersistBasicNetwork}
 * and only {@link #getPersistClassString()} is changed to 'BasicFloatNetwork'.
 * 
 * <p>
 * Because of all final methods in {@link PersistBasicNetwork}, we have to copy code while not take extension.
 */
public class PersistBasicFloatNetwork implements EncogPersistor {

    /**
     * {@inheritDoc}
     */
    @Override
    public final int getFileVersion() {
        return 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final String getPersistClassString() {
        return "BasicFloatNetwork";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final Object read(final InputStream is) {
        final BasicFloatNetwork result = new BasicFloatNetwork();
        final FlatNetwork flat = new FlatNetwork();
        final EncogReadHelper in = new EncogReadHelper(is);
        EncogFileSection section;
        while((section = in.readNextSection()) != null) {
            if(section.getSectionName().equals("BASIC") && section.getSubSectionName().equals("PARAMS")) {
                final Map<String, String> params = section.parseParams();
                result.getProperties().putAll(params);
            }
            if(section.getSectionName().equals("BASIC") && section.getSubSectionName().equals("NETWORK")) {
                final Map<String, String> params = section.parseParams();

                flat.setBeginTraining(EncogFileSection.parseInt(params, BasicNetwork.TAG_BEGIN_TRAINING));
                flat.setConnectionLimit(EncogFileSection.parseDouble(params, BasicNetwork.TAG_CONNECTION_LIMIT));
                flat.setContextTargetOffset(EncogFileSection.parseIntArray(params,
                        BasicNetwork.TAG_CONTEXT_TARGET_OFFSET));
                flat.setContextTargetSize(EncogFileSection.parseIntArray(params, BasicNetwork.TAG_CONTEXT_TARGET_SIZE));
                flat.setEndTraining(EncogFileSection.parseInt(params, BasicNetwork.TAG_END_TRAINING));
                flat.setHasContext(EncogFileSection.parseBoolean(params, BasicNetwork.TAG_HAS_CONTEXT));
                flat.setInputCount(EncogFileSection.parseInt(params, PersistConst.INPUT_COUNT));
                flat.setLayerCounts(EncogFileSection.parseIntArray(params, BasicNetwork.TAG_LAYER_COUNTS));
                flat.setLayerFeedCounts(EncogFileSection.parseIntArray(params, BasicNetwork.TAG_LAYER_FEED_COUNTS));
                flat.setLayerContextCount(EncogFileSection.parseIntArray(params, BasicNetwork.TAG_LAYER_CONTEXT_COUNT));
                flat.setLayerIndex(EncogFileSection.parseIntArray(params, BasicNetwork.TAG_LAYER_INDEX));
                flat.setLayerOutput(EncogFileSection.parseDoubleArray(params, PersistConst.OUTPUT));
                flat.setLayerSums(new double[flat.getLayerOutput().length]);
                flat.setOutputCount(EncogFileSection.parseInt(params, PersistConst.OUTPUT_COUNT));
                flat.setWeightIndex(EncogFileSection.parseIntArray(params, BasicNetwork.TAG_WEIGHT_INDEX));
                flat.setWeights(EncogFileSection.parseDoubleArray(params, PersistConst.WEIGHTS));
                flat.setBiasActivation(EncogFileSection.parseDoubleArray(params, BasicNetwork.TAG_BIAS_ACTIVATION));
            } else if(section.getSectionName().equals("BASIC") && section.getSubSectionName().equals("ACTIVATION")) {
                int index = 0;

                flat.setActivationFunctions(new ActivationFunction[flat.getLayerCounts().length]);

                for(final String line: section.getLines()) {
                    ActivationFunction af = null;
                    final List<String> cols = EncogFileSection.splitColumns(line);
                    String name = "org.encog.engine.network.activation." + cols.get(0);
                    if(cols.get(0).equals("ActivationReLU")) {
                        name = "ml.shifu.shifu.core.dtrain.nn.ActivationReLU";
                    }
                    try {
                        final Class<?> clazz = Class.forName(name);
                        af = (ActivationFunction) clazz.newInstance();
                    } catch (final ClassNotFoundException e) {
                        throw new PersistError(e);
                    } catch (final InstantiationException e) {
                        throw new PersistError(e);
                    } catch (final IllegalAccessException e) {
                        throw new PersistError(e);
                    }

                    for(int i = 0; i < af.getParamNames().length; i++) {
                        af.setParam(i, CSVFormat.EG_FORMAT.parse(cols.get(i + 1)));
                    }

                    flat.getActivationFunctions()[index++] = af;
                }
            } else if(section.getSectionName().equals("BASIC") && section.getSubSectionName().equals("SUBSET")) {
                final Map<String, String> params = section.parseParams();
                String subsetStr = params.get("SUBSETFEATURES");
                if(StringUtils.isBlank(subsetStr)) {
                    result.setFeatureSet(null);
                } else {
                    String[] splits = subsetStr.split(",");
                    Set<Integer> subFeatures = new HashSet<Integer>();
                    for(String split: splits) {
                        int featureIndex = Integer.parseInt(split);
                        subFeatures.add(featureIndex);
                    }
                    result.setFeatureSet(subFeatures);
                }
            }
        }

        result.getStructure().setFlat(flat);

        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public final void save(final OutputStream os, final Object obj) {
        final EncogWriteHelper out = new EncogWriteHelper(os);
        final BasicFloatNetwork net = (BasicFloatNetwork) obj;
        final FlatNetwork flat = net.getStructure().getFlat();
        out.addSection("BASIC");
        out.addSubSection("PARAMS");
        out.addProperties(net.getProperties());
        out.addSubSection("NETWORK");

        out.writeProperty(BasicNetwork.TAG_BEGIN_TRAINING, flat.getBeginTraining());
        out.writeProperty(BasicNetwork.TAG_CONNECTION_LIMIT, flat.getConnectionLimit());
        out.writeProperty(BasicNetwork.TAG_CONTEXT_TARGET_OFFSET, flat.getContextTargetOffset());
        out.writeProperty(BasicNetwork.TAG_CONTEXT_TARGET_SIZE, flat.getContextTargetSize());
        out.writeProperty(BasicNetwork.TAG_END_TRAINING, flat.getEndTraining());
        out.writeProperty(BasicNetwork.TAG_HAS_CONTEXT, flat.getHasContext());
        out.writeProperty(PersistConst.INPUT_COUNT, flat.getInputCount());
        out.writeProperty(BasicNetwork.TAG_LAYER_COUNTS, flat.getLayerCounts());
        out.writeProperty(BasicNetwork.TAG_LAYER_FEED_COUNTS, flat.getLayerFeedCounts());
        out.writeProperty(BasicNetwork.TAG_LAYER_CONTEXT_COUNT, flat.getLayerContextCount());
        out.writeProperty(BasicNetwork.TAG_LAYER_INDEX, flat.getLayerIndex());
        out.writeProperty(PersistConst.OUTPUT, flat.getLayerOutput());
        out.writeProperty(PersistConst.OUTPUT_COUNT, flat.getOutputCount());
        out.writeProperty(BasicNetwork.TAG_WEIGHT_INDEX, flat.getWeightIndex());
        out.writeProperty(PersistConst.WEIGHTS, flat.getWeights());
        out.writeProperty(BasicNetwork.TAG_BIAS_ACTIVATION, flat.getBiasActivation());
        out.addSubSection("ACTIVATION");
        for(final ActivationFunction af: flat.getActivationFunctions()) {
            out.addColumn(af.getClass().getSimpleName());
            for(int i = 0; i < af.getParams().length; i++) {
                out.addColumn(af.getParams()[i]);
            }
            out.writeLine();
        }
        out.addSubSection("SUBSET");
        Set<Integer> featureList = net.getFeatureSet();
        if(featureList == null || featureList.size() == 0) {
            out.writeProperty("SUBSETFEATURES", "");
        } else {
            String subFeaturesStr = StringUtils.join(featureList, ",");
            out.writeProperty("SUBSETFEATURES", subFeaturesStr);
        }
        out.flush();
    }

    public BasicFloatNetwork readNetwork(final DataInput in) throws IOException {
        final BasicFloatNetwork result = new BasicFloatNetwork();
        final FlatNetwork flat = new FlatNetwork();

        // read properties
        Map<String, String> properties = new HashMap<String, String>();
        int size = in.readInt();
        for(int i = 0; i < size; i++) {
            properties.put(ml.shifu.shifu.core.dtrain.StringUtils.readString(in),
                    ml.shifu.shifu.core.dtrain.StringUtils.readString(in));
        }
        result.getProperties().putAll(properties);

        // read fields
        flat.setBeginTraining(in.readInt());
        flat.setConnectionLimit(in.readDouble());

        flat.setContextTargetOffset(readIntArray(in));
        flat.setContextTargetSize(readIntArray(in));

        flat.setEndTraining(in.readInt());
        flat.setHasContext(in.readBoolean());
        flat.setInputCount(in.readInt());

        flat.setLayerCounts(readIntArray(in));
        flat.setLayerFeedCounts(readIntArray(in));
        flat.setLayerContextCount(readIntArray(in));
        flat.setLayerIndex(readIntArray(in));
        flat.setLayerOutput(readDoubleArray(in));
        flat.setOutputCount(in.readInt());
        flat.setLayerSums(new double[flat.getLayerOutput().length]);
        flat.setWeightIndex(readIntArray(in));
        flat.setWeights(readDoubleArray(in));
        flat.setBiasActivation(readDoubleArray(in));

        // read activations
        flat.setActivationFunctions(new ActivationFunction[flat.getLayerCounts().length]);
        int acSize = in.readInt();
        for(int i = 0; i < acSize; i++) {
            String name = ml.shifu.shifu.core.dtrain.StringUtils.readString(in);
            if(name.equals("ActivationReLU")) {
                name = ActivationReLU.class.getName();
            } else {
                name = "org.encog.engine.network.activation." + name;
            }
            ActivationFunction af = null;
            try {
                final Class<?> clazz = Class.forName(name);
                af = (ActivationFunction) clazz.newInstance();
            } catch (final ClassNotFoundException e) {
                throw new PersistError(e);
            } catch (final InstantiationException e) {
                throw new PersistError(e);
            } catch (final IllegalAccessException e) {
                throw new PersistError(e);
            }
            double[] params = readDoubleArray(in);
            for(int j = 0; j < params.length; j++) {
                af.setParam(j, params[j]);
            }
            flat.getActivationFunctions()[i] = af;
        }

        // read subset
        int subsetSize = in.readInt();
        Set<Integer> featureList = new HashSet<Integer>();
        for(int i = 0; i < subsetSize; i++) {
            featureList.add(in.readInt());
        }
        result.setFeatureSet(featureList);

        result.getStructure().setFlat(flat);
        return result;
    }

    public void saveNetwork(DataOutput out, final BasicFloatNetwork network) throws IOException {
        final FlatNetwork flat = network.getStructure().getFlat();
        // write general properties
        Map<String, String> properties = network.getProperties();
        if(properties == null) {
            out.writeInt(0);
        } else {
            out.writeInt(properties.size());
            for(Entry<String, String> entry: properties.entrySet()) {
                ml.shifu.shifu.core.dtrain.StringUtils.writeString(out, entry.getKey());
                ml.shifu.shifu.core.dtrain.StringUtils.writeString(out, entry.getValue());
            }
        }

        // write fields values in BasicFloatNetwork
        out.writeInt(flat.getBeginTraining());
        out.writeDouble(flat.getConnectionLimit());

        writeIntArray(out, flat.getContextTargetOffset());
        writeIntArray(out, flat.getContextTargetSize());

        out.writeInt(flat.getEndTraining());
        out.writeBoolean(flat.getHasContext());
        out.writeInt(flat.getInputCount());

        writeIntArray(out, flat.getLayerCounts());
        writeIntArray(out, flat.getLayerFeedCounts());
        writeIntArray(out, flat.getLayerContextCount());
        writeIntArray(out, flat.getLayerIndex());
        writeDoubleArray(out, flat.getLayerOutput());
        out.writeInt(flat.getOutputCount());
        writeIntArray(out, flat.getWeightIndex());
        writeDoubleArray(out, flat.getWeights());
        writeDoubleArray(out, flat.getBiasActivation());

        // write activation list
        out.writeInt(flat.getActivationFunctions().length);
        for(final ActivationFunction af: flat.getActivationFunctions()) {
            ml.shifu.shifu.core.dtrain.StringUtils.writeString(out, af.getClass().getSimpleName());
            writeDoubleArray(out, af.getParams());
        }
        // write sub sets
        Set<Integer> featureList = network.getFeatureSet();
        if(featureList == null || featureList.size() == 0) {
            out.writeInt(0);
        } else {
            out.writeInt(featureList.size());
            for(Integer integer: featureList) {
                out.writeInt(integer);
            }
        }
    }

    private int[] readIntArray(DataInput in) throws IOException {
        int size = in.readInt();
        int[] array = new int[size];
        for(int i = 0; i < size; i++) {
            array[i] = in.readInt();
        }
        return array;
    }

    private double[] readDoubleArray(DataInput in) throws IOException {
        int size = in.readInt();
        double[] array = new double[size];
        for(int i = 0; i < size; i++) {
            array[i] = in.readDouble();
        }
        return array;
    }

    private void writeIntArray(DataOutput out, int[] array) throws IOException {
        if(array == null) {
            out.writeInt(0);
        } else {
            out.writeInt(array.length);
            for(int i: array) {
                out.writeInt(i);
            }
        }
    }

    private void writeDoubleArray(DataOutput out, double[] array) throws IOException {
        if(array == null) {
            out.writeInt(0);
        } else {
            out.writeInt(array.length);
            for(double d: array) {
                out.writeDouble(d);
            }
        }
    }

}