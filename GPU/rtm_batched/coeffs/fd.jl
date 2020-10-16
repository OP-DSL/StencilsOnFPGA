using SymPy

#Julia script that generates finite difference coefficients
#to arbitrary precision


function lagrange(order)

  is=collect(-floor(Int,order/2) : 1 : floor(Int,order/2));
  x=symbols("x");
  L=x^0;
  Ls=Array(typeof(L),(order,));


  for j = 1 : order
    ith=is[j];
    for i in is
      if i!= ith
        L=L*(x-i)/(ith-i);
      end
    end
    Ls[j]=copy(L);
    L=x^0;
  end


  return Ls;

end



function fd(order,d;t=Rational)
  Ls=lagrange(order);
  x =symbols("x");

  is=collect(-floor(Int,order/2) : 1 : floor(Int,order/2));
  D=Array(typeof(x),(order,order));
  for i = 1 : order
    for j = 1 : order
      D[i,j] =diff(Ls[j],x,d)(x=>is[i]);
    end
  end


  return Array{t}(D);

end


for order = 2 : 2 : 16
  D=fd(order+1,1,t=Float64);
  f=open("coeffs$(order).h","w");
  write(f,"#ifndef _COEFFS_H_\n");
  write(f,"#define _COEFFS_H_\n");
  write(f,"#include \"../preface.h\"\n");
  write(f,"#define ORDER $(order)\n");
  write(f,"#define HALF  $(round(Int64,order/2))\n");
  write(f,"real_t coeffs[$(order+1)][$(order+1)] = ");
  (m,n)=size(D);
  write(f,"{\n");
  for i = 1 : m
    for j = 1 : n
      if j==1
        write(f," {");
      end
      if(j!=n)
        write(f,"$(D[i,j]),")
      else
        if i!=m
          write(f,"$(D[i,j])},")
        else
          write(f,"$(D[i,j])}\n")
        end
      end
    end
  end
  write(f,"\n};");
  write(f,"\n\n#endif");
  close(f);
end
