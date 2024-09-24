package dev.skyr.core.nn.layers;

import dev.skyr.core.autograd.Tensor;

import java.util.HashMap;

public class CustomModule extends Module{
    @Override
    public HashMap<String, Tensor> parameters() {

        HashMap<String, Tensor> params = new HashMap<>();
        for (java.lang.reflect.Field field : this.getClass().getDeclaredFields()) {
            if (Module.class.isAssignableFrom(field.getType())) {
                try {
                    field.setAccessible(true);
                    HashMap<String, Tensor> fieldParams = ((Module) field.get(this)).parameters();
                    for (String key : fieldParams.keySet()) {
                        params.put(field.getName() + "." + key, fieldParams.get(key));
                    }
                } catch (IllegalAccessException e) {
                    throw new RuntimeException(e);
                }
            }
        }
        return params;
    }
}
