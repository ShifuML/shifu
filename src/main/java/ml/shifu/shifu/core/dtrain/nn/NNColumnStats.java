/*
 * Copyright [2013-2017] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.nn;

import ml.shifu.guagua.io.Bytable;
import ml.shifu.shifu.container.obj.ColumnType;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * {@link NNColumnStats} is to wrapper valued info for neural network model serialization.
 */
public class NNColumnStats implements Bytable {

    public NNColumnStats() {
    }

    public NNColumnStats(int columnNum, String columnName, ColumnType columnType, double mean, double stddev,
            double woeMean, double woeStddev, double woeWgtMean, double woeWgtStddev, List<Double> binBoundaries,
            List<String> binCategories, List<Double> binPosRates, List<Double> binCountWoes, List<Double> binWeightWoes) {
        this.columnNum = columnNum;
        this.columnName = columnName;
        this.columnType = columnType;
        this.mean = mean;
        this.stddev = stddev;
        this.woeMean = woeMean;
        this.woeStddev = woeStddev;
        this.woeWgtMean = woeWgtMean;
        this.woeWgtStddev = woeWgtStddev;
        this.binBoundaries = binBoundaries;
        this.binCategories = binCategories;
        this.binPosRates = binPosRates;
        this.binCountWoes = binCountWoes;
        this.binWeightWoes = binWeightWoes;
    }

    private int columnNum;

    private String columnName;

    private ColumnType columnType;

    private double cutoff;

    private double mean;

    private double stddev;

    private double woeMean;

    private double woeStddev;

    private double woeWgtMean;

    private double woeWgtStddev;

    private List<Double> binBoundaries;

    private List<String> binCategories;

    private List<Double> binPosRates;

    private List<Double> binCountWoes;

    private List<Double> binWeightWoes;

    public boolean isCategorical() {
        return this.columnType == ColumnType.C;
    }
    
    public boolean isHybrid() {
        return this.columnType == ColumnType.H;
    }

    public boolean isNumerical() {
        return this.columnType == ColumnType.N || this.columnType == ColumnType.H;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.columnNum);
        ml.shifu.shifu.core.dtrain.StringUtils.writeString(out, this.columnName);
        out.writeByte(this.columnType.getByteType());
        out.writeDouble(this.cutoff);
        out.writeDouble(this.mean);
        out.writeDouble(this.stddev);
        out.writeDouble(this.woeMean);
        out.writeDouble(this.woeStddev);
        out.writeDouble(this.woeWgtMean);
        out.writeDouble(this.woeWgtStddev);

        writeDoubleList(out, this.binBoundaries);

        if(this.binCategories == null) {
            out.writeInt(0);
        } else {
            int cateSize = this.binCategories.size();
            out.writeInt(cateSize);
            for(int i = 0; i < cateSize; i++) {
                ml.shifu.shifu.core.dtrain.StringUtils.writeString(out, this.binCategories.get(i));
            }
        }

        writeDoubleList(out, this.binPosRates);
        writeDoubleList(out, this.binCountWoes);
        writeDoubleList(out, this.binWeightWoes);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.columnNum = in.readInt();
        this.columnName = ml.shifu.shifu.core.dtrain.StringUtils.readString(in);
        this.columnType = ColumnType.of(in.readByte());
        this.cutoff = in.readDouble();
        this.mean = in.readDouble();
        this.stddev = in.readDouble();
        this.woeMean = in.readDouble();
        this.woeStddev = in.readDouble();
        this.woeWgtMean = in.readDouble();
        this.woeWgtStddev = in.readDouble();

        this.binBoundaries = readDoubleList(in);

        int cateSize = in.readInt();
        List<String> categories = new ArrayList<String>(cateSize);
        for(int i = 0; i < cateSize; i++) {
            categories.add(ml.shifu.shifu.core.dtrain.StringUtils.readString(in));
        }

        this.binCategories = categories;
        this.binPosRates = readDoubleList(in);
        this.binCountWoes = readDoubleList(in);
        this.binWeightWoes = readDoubleList(in);
    }

    private List<Double> readDoubleList(DataInput in) throws IOException {
        int size = in.readInt();
        List<Double> list = new ArrayList<Double>();
        for(int i = 0; i < size; i++) {
            list.add(in.readDouble());
        }
        return list;
    }

    private void writeDoubleList(DataOutput out, List<Double> list) throws IOException {
        if(list == null) {
            out.writeInt(0);
        } else {
            out.writeInt(list.size());
            for(double d: list) {
                out.writeDouble(d);
            }
        }
    }

    /**
     * @return the columnType
     */
    public ColumnType getColumnType() {
        return columnType;
    }

    /**
     * @param columnType
     *            the columnType to set
     */
    public void setColumnType(ColumnType columnType) {
        this.columnType = columnType;
    }

    /**
     * @return the mean
     */
    public double getMean() {
        return mean;
    }

    /**
     * @param mean
     *            the mean to set
     */
    public void setMean(double mean) {
        this.mean = mean;
    }

    /**
     * @return the stddev
     */
    public double getStddev() {
        return stddev;
    }

    /**
     * @param stddev
     *            the stddev to set
     */
    public void setStddev(double stddev) {
        this.stddev = stddev;
    }

    /**
     * @return the woeMean
     */
    public double getWoeMean() {
        return woeMean;
    }

    /**
     * @param woeMean
     *            the woeMean to set
     */
    public void setWoeMean(double woeMean) {
        this.woeMean = woeMean;
    }

    /**
     * @return the woeStddev
     */
    public double getWoeStddev() {
        return woeStddev;
    }

    /**
     * @param woeStddev
     *            the woeStddev to set
     */
    public void setWoeStddev(double woeStddev) {
        this.woeStddev = woeStddev;
    }

    /**
     * @return the woeWgtMean
     */
    public double getWoeWgtMean() {
        return woeWgtMean;
    }

    /**
     * @param woeWgtMean
     *            the woeWgtMean to set
     */
    public void setWoeWgtMean(double woeWgtMean) {
        this.woeWgtMean = woeWgtMean;
    }

    /**
     * @return the woeWgtStddev
     */
    public double getWoeWgtStddev() {
        return woeWgtStddev;
    }

    /**
     * @param woeWgtStddev
     *            the woeWgtStddev to set
     */
    public void setWoeWgtStddev(double woeWgtStddev) {
        this.woeWgtStddev = woeWgtStddev;
    }

    /**
     * @return the binBoundaries
     */
    public List<Double> getBinBoundaries() {
        return binBoundaries;
    }

    /**
     * @param binBoundaries
     *            the binBoundaries to set
     */
    public void setBinBoundaries(List<Double> binBoundaries) {
        this.binBoundaries = binBoundaries;
    }

    /**
     * @return the binCategories
     */
    public List<String> getBinCategories() {
        return binCategories;
    }

    /**
     * @param binCategories
     *            the binCategories to set
     */
    public void setBinCategories(List<String> binCategories) {
        this.binCategories = binCategories;
    }

    /**
     * @return the binPosRates
     */
    public List<Double> getBinPosRates() {
        return binPosRates;
    }

    /**
     * @param binPosRates
     *            the binPosRates to set
     */
    public void setBinPosRates(List<Double> binPosRates) {
        this.binPosRates = binPosRates;
    }

    /**
     * @return the binCountWoes
     */
    public List<Double> getBinCountWoes() {
        return binCountWoes;
    }

    /**
     * @param binCountWoes
     *            the binCountWoes to set
     */
    public void setBinCountWoes(List<Double> binCountWoes) {
        this.binCountWoes = binCountWoes;
    }

    /**
     * @return the binWeightWoes
     */
    public List<Double> getBinWeightWoes() {
        return binWeightWoes;
    }

    /**
     * @param binWeightWoes
     *            the binWeightWoes to set
     */
    public void setBinWeightWoes(List<Double> binWeightWoes) {
        this.binWeightWoes = binWeightWoes;
    }

    /**
     * @return the columnName
     */
    public String getColumnName() {
        return columnName;
    }

    /**
     * @param columnName
     *            the columnName to set
     */
    public void setColumnName(String columnName) {
        this.columnName = columnName;
    }

    /**
     * @return the columnNum
     */
    public int getColumnNum() {
        return columnNum;
    }

    /**
     * @param columnNum
     *            the columnNum to set
     */
    public void setColumnNum(int columnNum) {
        this.columnNum = columnNum;
    }

    /**
     * @return the cutoff
     */
    public double getCutoff() {
        return cutoff;
    }

    /**
     * @param cutoff
     *            the cutoff to set
     */
    public void setCutoff(double cutoff) {
        this.cutoff = cutoff;
    }

}
