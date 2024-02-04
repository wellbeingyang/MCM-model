import numpy as np
import tools
def step_forward(t):
    #先进行速度加权平均
    v_a=np.array([0,0,0])
    v_a[0]=tools.p_disable(t)*tools.v_before[0]
    v_a[1]=tools.p_disable(t)*tools.v_before[1]
    v_a[2]=tools.p_disable(t)*tools.v_before[2]+(1-tools.p_disable(t))*tools.VZ

    #再进行位置、洋流速度、密度加权平均,并用这些数据求力
    p_a=np.array([0,0,0])
    v_current_a=np.array([0,0,0])
    dens_a=np.sum(tools.P*tools.density_water)
    P_expanded=tools.P[:,:,:,np.newaxis]
    v_current_a=np.sum(P_expanded*tools.current_v,axis=(0,1,2))
    p_a[0]=np.sum(np.sum(tools.P,axis=(1,2)) * np.arange(tools.shape[0]))
    p_a[1]=np.sum(np.sum(tools.P,axis=(0,2)) * np.arange(tools.shape[1]))
    p_a[2]=np.sum(np.sum(tools.P,axis=(0,1)) * np.arange(tools.shape[2]))
    force=np.array([0,0,0])
    force=tools.update_force_prediction(p_a,v_a,tools.k,tools.mass,tools.g,tools.density,dens_a,v_current_a,tools.height)

    #进行求新P的操作
    P_new=tools.convolve_probability_density(tools.P,v_a)
    tools.P=P_new

    #同时需要给出新速度
    #求新速度方法：用老位置（全空间加权的）、老速度（是否失去动力加权的）算出新位置，再用新位置、老速度、老力算出新速度
    position_new=tools.update_position(p_a,v_a)
    position_new=tools.check_position(position_new)
    v_new=tools.update_speed_prediction(position_new,v_a,tools.height,force)
    tools.v_before=v_new