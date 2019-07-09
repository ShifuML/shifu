package ml.shifu.shifu.core.dtrain.multitask;

import ml.shifu.guagua.io.Bytable;
import ml.shifu.guagua.io.Combinable;
import ml.shifu.shifu.core.dtrain.SerializationType;
import ml.shifu.shifu.core.dtrain.wdl.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public class MultiTaskNN implements WeightInitializer<MultiTaskNN>, Bytable, Combinable<MultiTaskNN> {

    private static final Logger LOG = LoggerFactory.getLogger(MultiTaskNN.class);

    private DenseInputLayer dil;

    private List<Layer> hiddenLayers;

    private DenseLayer finalLayer;

    private Map<Integer, Integer> idBinCateSizeMap;

    private int numericalSize;

    private List<Integer> hiddenNodes;

    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    public MultiTaskNN() {
    }

    public MultiTaskNN(DenseInputLayer dil, List<Layer> hiddenLayers, DenseLayer finalLayer, Map<Integer, Integer> idBinCateSizeMap,
                       int numericalSize, List<Integer> hiddenNodes, SerializationType serializationType) {
        this.dil = dil;
        this.hiddenLayers = hiddenLayers;
        this.finalLayer = finalLayer;
        this.idBinCateSizeMap = idBinCateSizeMap;
        this.numericalSize = numericalSize;
        this.hiddenNodes = hiddenNodes;
        this.serializationType = serializationType;
    }

    public float[] forward(){
        return null;
    }

    public static Logger getLOG() {
        return LOG;
    }

    public DenseInputLayer getDil() {
        return dil;
    }

    public void setDil(DenseInputLayer dil) {
        this.dil = dil;
    }

    public List<Layer> getHiddenLayers() {
        return hiddenLayers;
    }

    public void setHiddenLayers(List<Layer> hiddenLayers) {
        this.hiddenLayers = hiddenLayers;
    }

    public DenseLayer getFinalLayer() {
        return finalLayer;
    }

    public void setFinalLayer(DenseLayer finalLayer) {
        this.finalLayer = finalLayer;
    }

    public Map<Integer, Integer> getIdBinCateSizeMap() {
        return idBinCateSizeMap;
    }

    public void setIdBinCateSizeMap(Map<Integer, Integer> idBinCateSizeMap) {
        this.idBinCateSizeMap = idBinCateSizeMap;
    }

    public int getNumericalSize() {
        return numericalSize;
    }

    public void setNumericalSize(int numericalSize) {
        this.numericalSize = numericalSize;
    }

    public List<Integer> getHiddenNodes() {
        return hiddenNodes;
    }

    public void setHiddenNodes(List<Integer> hiddenNodes) {
        this.hiddenNodes = hiddenNodes;
    }

    public SerializationType getSerializationType() {
        return serializationType;
    }

    public void setSerializationType(SerializationType serializationType) {
        this.serializationType = serializationType;
    }

    @Override
    public void initWeight(InitMethod method) {

    }

    @Override
    public void initWeight(MultiTaskNN updateModel) {

    }


    @Override
    public void write(DataOutput out) throws IOException {

    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }

    @Override
    public MultiTaskNN combine(MultiTaskNN from) {
        return null;
    }
}
