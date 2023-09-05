class basenode {
    constructor(){
        this.w = Math.random();
        this.f_rounds=0;
        this.b_rounds=0;
    }
    forward(x){
        this.x = x;
        if(x<0){
            this.z = 0
        }else{
            this.z = x*w
        }        
        this.f_rounds++;
        
    }


}