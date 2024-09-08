import { ref, computed, reactive } from 'vue'
import { defineStore } from 'pinia'

export const userDataStore = defineStore('userData', {
    state: () =>({
      username:ref("xxx"),
      phone:ref("xxx"),
      email:ref("xxx"),
      isRumor:ref(100),
      notRumor:ref(50),
      rumorList:ref<Array<any>>([
        {
          "id" : "null",
          "isRumor" : "null",
          "rumorText" : "null"
        }
      ]),
    }),
    getters:{
      getUsername() : string{
        return this.username;
      },
      getPhone() : string{
        return this.phone;
      },
      getEmail() : string{
        return this.email;
      },
      getIsRumor() : number{
        return this.isRumor;
      },
      getNotRumor() : number{
        return this.notRumor;
      },
      getRumorList() : Array<any>{
        return this.rumorList;
      }
    },
    actions: {
      setUsername(username : string){
        this.username = username;
      },
      setPhone(phone : string){
        this.phone = phone;
      },
      setEmail(email : string){
        this.email = email;
      },
      setIsRumor(isRumor : number){
        this.isRumor = isRumor;
      },
      setNotRumor(notRumor : number){
        this.notRumor = notRumor;
      },
      setRumorList(rumorList : Array<any>){
        this.rumorList = rumorList;
      }
    }
  })