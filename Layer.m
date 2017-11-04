classdef Layer < handle
   properties
      layerOrdinal = 0;%will be set in ctor
%       totalLayers = 0;%will be set in ctor
      neigh = [];%neigh(1) = prev layer. neigh(2) = next layer
      x = [];%inputs [x1 x2 x3 ...]
      w = [];%weights [[w11;w21] [w12;w22] ...]
      deltaW = [];%weight change
      desiredOutput = [];
      lg = 0.1;%learning gain
      mg = 0.9;%momentum gain
      finalOp = [];%Also stores final output
      %The second last layer is actually the last layer
      isSecondLastLayer = 0;%can be overwritten during NN construction
      activationFunction = 'sigmoid';%default. Can be overridden
   end
   
   methods
       function this = Layer(numInputs, numOutputs, ord, activ)%ctor
           this.x = zeros(1, numInputs);
           this.w = 0.2*rand(numOutputs, numInputs)-0.1;
           this.deltaW = zeros(size(this.w));
           this.layerOrdinal = ord;
%            this.totalLayers = tot;
           this.activationFunction = activ;
       end
       
       function delta = Train(this, inp)
           %fprintf('Training: Layer%d\n', this.layerOrdinal);
           if this == this.neigh(2),%----LAST LAYER. compute error                       
               delta = this.desiredOutput - inp;
               this.finalOp = inp;
           else                     %----OTHER LAYERS
               %--- F E E D  F O R W A R D
               y = this.w * inp;
               if ~this.isSecondLastLayer,
                   y = this.activation(y);
               end
               deltaNext = this.neigh(2).Train(y);               
               
               %--- B A C K P R O P A G A T I O N
               if this ~= this.neigh(1),%if not first layer
                ed = inp;
                for i=1:length(inp)
                    ed(i) = this.w(:,i)' * deltaNext;
                end
                delta = inp .* (1-inp).*ed;
               end
               this.deltaW = this.lg.*deltaNext*inp' + this.mg.*this.deltaW;  
           end
       end
       
       function UpdateWeights(this)
           if this ~= this.neigh(2),%if not last layer
               this.w = this.w + this.deltaW;             
               this.neigh(2).UpdateWeights();
           end
       end
       
       function Predict(this, inp)
           if this == this.neigh(2),%if last layer                        
                this.finalOp = inp;
           else               
               y= this.w * inp;               
               if ~this.isSecondLastLayer,
                   y = this.activation(y);
               end
               this.neigh(2).Predict(y);%Pass on output to next layer               
           end
       end
       
       function setNeighbours(this, prev, next)
%            if prev == this, disp('first node');end
%            if next == this, disp('last node');end
           this.neigh = [prev next];
       end
       
       function setDesiredOutput(this, desOp)
           this.desiredOutput = desOp;
       end
       
       function op = getOutput(this)
           op = this.finalOp;
       end
       
       function showWeights(this)
           fprintf('Layer %d weights\n', this.layerOrdinal);
           disp(this.w);
           if this.neigh(2) ~= this, this.neigh(2).showWeights();end
       end
       
       function setAsSecondLastLayer(this)
           this.isSecondLastLayer = 1;
       end
       
       function y = activation(this, y)
           if strcmp(this.activationFunction, 'sigmoid')
               y = 1./(1+exp(-y));
           end
           if strcmp(this.activationFunction, 'unipolar')
               %to be programmed
           end           
       end

   end
end
