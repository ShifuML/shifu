package ml.shifu.shifu.pmml.obj;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.dataformat.xml.annotation.JacksonXmlElementWrapper;
import com.fasterxml.jackson.dataformat.xml.annotation.JacksonXmlProperty;

import java.util.List;

public class DataDictionary {

    @JacksonXmlProperty(isAttribute=true)
    private int numberOfFields;

    private List<DataField> dataFields;

    public int getNumberOfFields() {
        return numberOfFields;
    }

    public void setNumberOfFields(int numberOfFields) {
        this.numberOfFields = numberOfFields;
    }

    public List<DataField> getDataFields() {
        return dataFields;
    }

    @JacksonXmlElementWrapper(useWrapping=false)
    @JacksonXmlProperty(localName="DataField")
    public void setDataFields(List<DataField> dataFields) {
        this.dataFields = dataFields;
    }

    public static class DataField {

        @JacksonXmlProperty(isAttribute=true)
        private String dataType;

        @JacksonXmlProperty(isAttribute=true)
        private String name;

        @JacksonXmlProperty(isAttribute=true)
        private String displayName;

        @JacksonXmlProperty(isAttribute=true)
        private String optype;

        private List<Value> values;
        private List<Interval> intervals;

        public String getDataType() {
            return dataType;
        }

        public void setDataType(String dataType) {
            this.dataType = dataType;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String getDisplayName() {
            return displayName;
        }

        public void setDisplayName(String displayName) {
            this.displayName = displayName;
        }

        public String getOptype() {
            return optype;
        }

        public void setOptype(String optype) {
            this.optype = optype;
        }

        public List<Value> getValues() {
            return values;
        }

        @JacksonXmlElementWrapper(useWrapping=false)
        @JacksonXmlProperty(localName="Value")
        public void setValue(List<Value> values) {
            this.values = values;
        }

        public List<Interval> getIntervals() {
            return intervals;
        }

        @JacksonXmlElementWrapper(useWrapping=false)
        @JacksonXmlProperty(localName="Interval")
        public void setInterval(List<Interval> intervals) {
            this.intervals = intervals;
        }


    }

    public static class Value {
        private String value;
        private String displayValue;
        private String property;

        public String getDisplayValue() {
            return displayValue;
        }

        public void setDisplayValue(String displayValue) {
            this.displayValue = displayValue;
        }

        public String getValue() {
            return value;
        }

        public void setValue(String value) {
            this.value = value;
        }

        public String getProperty() {
            return property;
        }

        public void setProperty(String property) {
            this.property = property;
        }


    }

    public static class Interval {
        private String closure;
        private String leftMargin;
        private String rightMargin;

        public String getClosure() {
            return closure;
        }

        public void setClosure(String closure) {
            this.closure = closure;
        }

        public String getLeftMargin() {
            return leftMargin;
        }

        public void setLeftMargin(String leftMargin) {
            this.leftMargin = leftMargin;
        }

        public String getRightMargin() {
            return rightMargin;
        }

        public void setRightMargin(String rightMargin) {
            this.rightMargin = rightMargin;
        }


    }

}


