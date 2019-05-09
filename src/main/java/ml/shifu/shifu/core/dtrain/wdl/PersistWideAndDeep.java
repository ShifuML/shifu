/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.wdl;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.dtrain.AssertUtils;
import ml.shifu.shifu.core.dtrain.wdl.activation.ReLU;
import ml.shifu.shifu.core.dtrain.wdl.activation.Sigmoid;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Persistent WideAndDeep models.
 * <p>
 * In this class, we don't close the input or output stream. So you need to close the stream where you open it.
 * <p>
 * @author  Wu Devin (haifwu@paypal.com)
 */
public class PersistWideAndDeep {

    /**
     * Save the WideAndDeep into output stream.
     * @param wnd, the WideAndDeep model
     * @param dos, the data output stream
     * @throws IOException IOException when IO operation
     */
    public static void save(final WideAndDeep wnd, DataOutputStream dos) throws IOException {
        dos.writeUTF(WideAndDeep.class.getName());

        // Only write DenseLayer in hidden layers
        List<DenseLayer> denseLayers = getAllDenseLayers(wnd.getHiddenLayers());
        writeList(denseLayers, dos);
        dos.writeInt(wnd.getNumericalSize());
        writeDenseLayer(wnd.getFinalLayer(), dos);
        writeEmbedLayer(wnd.getEcl(), dos);
        writeWideLayer(wnd.getWl(), dos);
        writeIntegerMap(wnd.getIdBinCateSizeMap(), dos);
        writeList(wnd.getDenseColumnIds(), dos);
        writeList(wnd.getEmbedColumnIds(), dos);
        writeList(wnd.getEmbedOutputs(), dos);
        writeList(wnd.getWideColumnIds(), dos);
        writeList(wnd.getHiddenNodes(), dos);
        writeList(wnd.getActiFuncs(), dos);

        dos.writeFloat(wnd.getL2reg());
    }

    /**
     * Load the WideAndDeep from input stream.
     * @param dis, the data input stream
     * @return the WideAndDeep model
     * @throws IOException IOException when IO operatio
     */
    @SuppressWarnings("rawtypes")
    public static WideAndDeep load(DataInputStream dis) throws IOException {
        AssertUtils.assertEquals(dis.readUTF(), WideAndDeep.class.getName());

        // Read DensorLayer only
        List<DenseLayer> denseLayers = readList(dis, DenseLayer.class);
        int numericalSize = dis.readInt();
        DenseLayer finalLayer = readDenseLayer(dis);
        EmbedLayer ecl = readEmbedLayer(dis);
        WideLayer wl = readWideLayer(dis);
        Map<Integer, Integer> idBinCateSizeMap = readIntegerMap(dis);
        List<Integer> denseColumnIds = readList(dis, Integer.class);
        List<Integer> embedColumnIds = readList(dis, Integer.class);
        List<Integer> embedOutputs = readList(dis, Integer.class);
        List<Integer> wideColumnIds = readList(dis, Integer.class);
        List<Integer> hiddenNodes = readList(dis, Integer.class);
        List<String> actiFuncs = readList(dis, String.class);
        float l2reg = dis.readFloat();

        List<Layer> hiddenLayers = buildHiddenLayers(denseLayers, actiFuncs);
        return new WideAndDeep(hiddenLayers, finalLayer, ecl, wl, idBinCateSizeMap, numericalSize, denseColumnIds,
                embedColumnIds, embedOutputs, wideColumnIds, hiddenNodes, actiFuncs, l2reg);
    }

    /**
     * Write list of String, Integer, ColumnConfig to output stream
     * <p>
     * Only support three type of data now: String, Integer, ColumnConfig
     * <p>
     * @param list, the list write to output stream
     * @param dos, data output stream
     * @param <T>, generic type
     * @throws IOException IOException when IO operation
     */
    private static <T> void writeList(List<T> list, DataOutputStream dos) throws IOException {
        dos.writeInt(list.size());
        for (T element: list) {
            if (element instanceof Integer) {
                dos.writeInt((Integer) element);
            } else if (element instanceof String) {
                dos.writeUTF((String) element);
            } else if (element instanceof ColumnConfig) {
                writeColumnConfig((ColumnConfig) element, dos);
            } else if (element instanceof DenseLayer) {
                writeDenseLayer((DenseLayer) element, dos);
            } else if (element instanceof EmbedFieldLayer) {
                writeEmbedFieldLayer((EmbedFieldLayer) element, dos);
            } else if (element instanceof WideFieldLayer) {
                writeWideFieldLayer((WideFieldLayer) element, dos);
            }
        }
    }

    /**
     * Read list of String, Integer, ColumnConfig from input stream
     * <p>
     * Only support three type of data now: String, Integer, ColumnConfig
     * <p>
     * @param dis, the data input stream
     * @param tClass, the class type of the object in the list
     * @param <T>, generic type
     * @return A list of specific object
     * @throws IOException IOException when IO operation
     */
    @SuppressWarnings("unchecked") 
    private static <T> List<T> readList(DataInputStream dis, Class<?> tClass) throws IOException {
        int size = dis.readInt();
        List<T> list = new ArrayList<>(size);
        for(int i = 0; i < size; i++){
            if (tClass == Integer.class) {
                list.add((T) Integer.valueOf(dis.readInt()));
            } else if (tClass == String.class) {
                list.add((T) dis.readUTF());
            } else if (tClass == ColumnConfig.class) {
                list.add((T) readColumnConfig(dis));
            } else if (tClass == DenseLayer.class) {
                list.add((T) readDenseLayer(dis));
            } else if (tClass == EmbedFieldLayer.class) {
                list.add((T) readEmbedFieldLayer(dis));
            } else if (tClass == WideFieldLayer.class) {
                list.add((T) readWideFieldLayer(dis));
            }
        }
        return list;
    }

    private static void writeIntegerMap(Map<Integer, Integer> map, DataOutputStream dos) throws IOException {
        dos.writeInt(map.size());
        for(Map.Entry<Integer, Integer> entry : map.entrySet()) {
            dos.writeInt(entry.getKey());
            dos.writeInt(entry.getValue());
        }
    }

    private static Map<Integer, Integer> readIntegerMap(DataInputStream dis) throws IOException {
        int size = dis.readInt();
        Map<Integer, Integer> map = new HashMap<>(size);
        for(int i = 0; i < size; i++) {
            map.put(dis.readInt(), dis.readInt());
        }
        return map;
    }

    private static void writeColumnConfig(ColumnConfig element, DataOutputStream dos) throws IOException {
        element.write(dos);
    }

    private static ColumnConfig readColumnConfig(DataInputStream dis) throws IOException {
        ColumnConfig columnConfig = new ColumnConfig();
        columnConfig.read(dis);
        return columnConfig;
    }

    private static void writeDenseLayer(DenseLayer denseLayer, DataOutputStream dos) throws IOException {
        dos.writeInt(denseLayer.getIn());
        dos.writeInt(denseLayer.getOut());
        // write bias
        for(int i = 0; i < denseLayer.getOut(); i++) {
            dos.writeFloat(denseLayer.getBias()[i]);
        }
        // write weight
        for(int i = 0; i < denseLayer.getIn(); i++) {
            for(int j = 0; j < denseLayer.getOut(); j++) {
                dos.writeFloat(denseLayer.getWeights()[i][j]);
            }
        }
        dos.writeFloat(denseLayer.getL2reg());
    }

    private static DenseLayer readDenseLayer(DataInputStream dis) throws IOException {
        int in = dis.readInt();
        int out = dis.readInt();
        // read bias
        float[] bias = new float[out];
        for(int i = 0; i < out; i++){
            bias[i] = dis.readFloat();
        }
        // read weight
        float[][] weights = new float[in][out];
        for(int i = 0; i < in; i++) {
            for(int j = 0; j < out; j++) {
                weights[i][j] = dis.readFloat();
            }
        }
        float l2reg = dis.readFloat();
        return new DenseLayer(weights, bias, out, in, l2reg);
    }

    private static void writeEmbedLayer(EmbedLayer embedLayer, DataOutputStream dos) throws IOException {
        writeList(embedLayer.getEmbedLayers(), dos);
    }

    private static EmbedLayer readEmbedLayer(DataInputStream dis) throws IOException {
        List<EmbedFieldLayer> embedFieldLayers = readList(dis, EmbedFieldLayer.class);
        return new EmbedLayer(embedFieldLayers);
    }

    private static void writeEmbedFieldLayer(EmbedFieldLayer embedFieldLayer, DataOutputStream dos) throws IOException {
        dos.writeInt(embedFieldLayer.getColumnId());
        dos.writeInt(embedFieldLayer.getIn());
        dos.writeInt(embedFieldLayer.getOut());
        // write weight
        for (int i = 0; i < embedFieldLayer.getIn(); i++){
            for (int j = 0; j < embedFieldLayer.getOut(); j++) {
                dos.writeFloat(embedFieldLayer.getWeights()[i][j]);
            }
        }
    }

    private static EmbedFieldLayer readEmbedFieldLayer(DataInputStream dis) throws IOException {
        int columnId = dis.readInt();
        int in = dis.readInt();
        int out = dis.readInt();
        // read weight
        float[][] weights = new float[in][out];
        for(int i = 0; i < in; i++) {
            for(int j = 0; j < out; j++) {
                weights[i][j] = dis.readFloat();
            }
        }
        return new EmbedFieldLayer(columnId, weights, out, in);
    }

    private static void writeWideLayer(WideLayer wideLayer, DataOutputStream dos) throws IOException {
        // write wide field layer list
        writeList(wideLayer.getLayers(), dos);
        // write bias layer
        writeBiasLayer(wideLayer.getBias(), dos);
    }

    private static WideLayer readWideLayer(DataInputStream dis) throws IOException {
        List<WideFieldLayer> wideFieldLayers = readList(dis, WideFieldLayer.class);
        BiasLayer biasLayer = readBiasLayer(dis);
        return new WideLayer(wideFieldLayers, biasLayer);
    }

    private static void writeWideFieldLayer(WideFieldLayer wideFieldLayer, DataOutputStream dos) throws IOException {
        dos.writeInt(wideFieldLayer.getColumnId());
        dos.writeInt(wideFieldLayer.getIn());
        // write weight
        for(int i = 0; i < wideFieldLayer.getIn(); i++){
            dos.writeFloat(wideFieldLayer.getWeights()[i]);
        }
    }

    private static WideFieldLayer readWideFieldLayer(DataInputStream dis) throws IOException {
        int columnId = dis.readInt();
        int in = dis.readInt();
        // read weight
        float[] weights = new float[in];
        for(int i = 0; i < in; i++) {
            weights[i] = dis.readFloat();
        }
        float l2reg = dis.readFloat();
        return new WideFieldLayer(columnId, weights, in, l2reg);
    }

    private static void writeBiasLayer(BiasLayer bias, DataOutputStream dos) throws IOException {
        dos.writeFloat(bias.getWeight());
    }

    private static BiasLayer readBiasLayer(DataInputStream dis) throws IOException {
        float weight = dis.readFloat();
        return new BiasLayer(weight);
    }

    @SuppressWarnings("rawtypes")
    private static List<DenseLayer> getAllDenseLayers(List<Layer> hiddenLayers) {
        AssertUtils.assertNotNull(hiddenLayers);

        List<DenseLayer> denseLayers = new ArrayList<>(hiddenLayers.size() / 2);
        for(Layer<?, ?, ?, ?> layer: hiddenLayers) {
            if (layer instanceof DenseLayer) {
                denseLayers.add((DenseLayer) layer);
            }
        }
        return denseLayers;
    }

    @SuppressWarnings("rawtypes")
    private static List<Layer> buildHiddenLayers(List<DenseLayer> denseLayers, List<String> actiFuncs){
        AssertUtils.assertListNotNullAndSizeEqual(denseLayers, actiFuncs);
        List<Layer> hiddenLayers = new ArrayList<>(actiFuncs.size() * 2);

        for(int i = 0; i < denseLayers.size(); i++) {
            hiddenLayers.add(denseLayers.get(i));

            String acti = actiFuncs.get(i);
            if("relu".equalsIgnoreCase(acti)) {
                hiddenLayers.add(new ReLU());
            } else if("sigmoid".equalsIgnoreCase(acti)) {
                hiddenLayers.add(new Sigmoid());
            }
        }

        return hiddenLayers;
    }

}
