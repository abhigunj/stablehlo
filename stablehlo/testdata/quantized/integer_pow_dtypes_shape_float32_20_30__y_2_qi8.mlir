// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<20x30xf32> {
    %cst = stablehlo.constant dense<"0xC578AB404A974C406DCE91BFE77CC0BF019117402F7FC7BF0AC8F3C07A6E1AC09FEE573E4B2489BEFC43AF4088DD2440E977B0C0758CACBEAB1A2ABFA7A02CC0151EFDC0627D2E409B7225BE7E106A4052A526BE5FF14AC078C69140047BD3BEF9EE8E3F17AF3D40F0C395BE3DA802C00DC304C03A17C3401CF7E0BDB09C513FF7C02040EB0F7C4023228FBFF0C0BCBEAEF1A0C038917A401C0012407E9AD1BFA030B8C0FB35C73D07A15CC0E1E301C0F93699C0FEFA4BBE32217C407DEDBEBF7FB343C0327D03C0B7A6AB40D72A28C0491B53C0F3FA46BF3E19F9BFBA7BA9BF673815401445ED3FE9010EC011B63940BCB627BE38D7D7BE7296CE3F02EDA940B293873F15D08EBFE176FCBCF7EA55C0893A11405E14AA3F16151EC000DD92C08F10A3C028FA534001AEB8BF55C1A1BC3235B940288982401548A93E91CFDEBF6BB735C0F94BC33FD79914C06AA363C02ACE5DC07D6CA3BF93ACEF3FE5260EC0BD0667C04460FD3EB67E8F3FD327C13E3184D8C0A41AB6BC56F3EFBF94748F408D938040B085193F2A5F43C0EDD26740F040A43F8CD9873FBA5301405EAA7BC061DC8C406D7F1640B575A0400566763E9FB8A13E33E5184098210FC0AC6E43BF9C9816C06D7345406590834029512CC0B6C88F3EAAE3F63F925D0040EF68113F4112593F1A32C140003E48C05C3A873F6C6B5240EC902A402C7DA53FB1568C402DD820C0B801843FE6771F407CA0153D360309401BE1A73E09DA7CBFC7C542C07DD4A1BE8C4BA840B58822407FB20440D748663E5228ADC0775B7E4095D3D3BD9F70B73F76623C40E7CC31C06A32D63F81718E3F02FD00BF5FFF3740A137D7C07106F13F8A93C5BED8ED79BF4D13CFBF763DC74013F390BE9FF7BD3E4EF6C7BEDD5037C0FDAF5B40EEF2D240AD04D8BFC6F700BFBD3D0540E2938740209845BEF64AD33F042D683F093AA3BF4592D23FC4410FBF8DB83CBE8BD68640E06261C02F219A406BF7773F84307DC0611C7C4096A5B240ADE20BC012B588C0C91FA7C068FDA140FF7A0A40B98B9EC07C7CFF3FFA2183C0E639C93F5719A0C029BEFB3C41D6264000E799402ECB9D3F454F534058FC63BDC9483140D8CBE43F3DA878401DB33840BCF26EC0C0C4B6C0C62B2E3FAB4688BF6710943F8C852D40BC4EAE40460FD33E16BC6540903D2EBF61D6FEBF07FC9BC033A66BC099A81AC00E2CD6BCA31155BD44440440C47CE03FCEAF59401099F5BEFD425E40A5C408400E0F37BED033C73E30DB84C0EBD3A9BEC17FC23FBF5260BE6AD6843F0A1B95BF22049A3E05B92A40C724DEBEACED4A3FCCE257BEA53A4B4005DE43404448F63F202ACBBF0490FCBE4F8D2440E417A0406DD40441F938EF3F12E8C43F633AC7C067D120BFA2BB33C02FB252BF31CD11C018C554C061836BC0147CBC3F8F4FA33F3F673BBEB2A3C63F49E23EBFC91239BF0CD8FC404694963E105568C02F970E3FF662C33F2750D13E2C11C5C0974465C0CF8CE4BF04A73640D412E5BFB629013E93C404C0D54427C07849D03FD80B97BFBE1259C00430DD3E9F6F51C095931E3F21E292C00300C5C044021FC04F33833F01CF5140769383BFD04A41400E5C9E40C88C12C0D992BB3F057557C0C81735405FECD43F8F4A0C3FF50912C0D0424EC01B2ED640446FC4BFF5A2A63F9621ACBE15FAA14089A04B3FE1BF99BF886A0EC0FBA01F4043859AC0FD7F1740405479C062DCCC3FEBCEDA3F90220FBF06AC814043D38FBFD20C4A3FD0280540406099C0FB0DE2409C8F0DBFFD40B7BF7F5F124074FA27C00A2B76C024B88E3D31A52340FE5D33C02A8A85C0A2053B40EC9E8340924179BF461EB3BEEC252440F73D9D3F1065333FA3E55F403E58B6402F01A2BD37DFDCBF9B6037C0E1502C40BDAE9A3FC8364640DFCC1BBFEBEE8BC04CC179400459FA3C6A55A5BF6C476BC0479AC9C0520E893F572B823FBF838FC0931C034025E5F6BFD71C3C3FCF0CA540142BAC3D7C59353F6700B53F44B964C0D11B823EB9AC093F57E2FB3D1703D53F5934073FA91502BFA032BDBFF06D1F4057DF6A40A6B292BF3A8296404B97DABFB3028E3F76FDF3BC4F0F3B402864CA40B6099FBF511E244067FC2B3ECE5A4FC0A026CF40151B03C019AE2E40D1743EBE2A9370402FA684C0D7601DC068D99340343CDE3F1A684840805FB0406455F83F51AA1EC09F9B833F872190C0286ED7BFA3DC8140805FABC01E46F7BF6D8062C0EE8F72405EAB643E4B6EFDBF19C08E3F1373DBBE1F2A11419C1E7F3F1DBC7FBFB63B86BEC0EABBC05B7CC740FE03BAC0B694494067ADFB3F7AE010409E13AF3EFA717640BD79E4C044103240D4B228BFCADBBCC0093519C0A6A1F53F8A451EC0557E9BBF535CE6BF89703540D5414BBFFA702CC0628482C03D9161C0DD0569C0C9B19D400C70BABF35D3B83EDEC005C1508A2DC02D3483C045C4BC40FB12DC40FD099540F4A2AE3DA08A1840B5A4073F676DC4BFB199043F2516F33F9C2EA74005ED3040127157C0891E81BF09AA3E4015229640DB5E2640210BBD3F55A816C0AE3680402BD77740FBE9A33F550F4BC01D038C40B9CA84C0C7660BC049E5903EA12578BF891D013FFFB016BF8491AF3E1FB0FA401AECC13EC98C33C02FD9CDBF019CD03F0E086E3F3F1A80C05730AFC08AD36E4074CBD0C0335F8AC07A49ED3F2816C63E3A8302407B0D2C4024BAAD3E481D8940E4949F4081D6CA3F31BF98BF9105C6C009F47BC0E7437BBE9B258B402BA033C08833AABE4F002E40FBC8EDBE606555C067AD28C01F455DBFBCD2733FBDABB9BF1F30CC3F9063B0BF57E2BF4013B28F40C19FBAC095626C3E23432F40F1F033409D1137BF7FFC7A408FD4814053B973C07FEE25C086FB8F3F811E67BE5EB41F405BDA58BFDD720B40E3FAB5BF72431DC00D6E13C0880B273F0A47ACC03524993E132F6F4032A60EC070108240DB21BD404FE4AA3F10B0923FA5918A40C517AFBF01B48DBE1845133FF87C95BF202BC73F4CFEED3DEAA5333FEF7A1BC055548A3FEF6E84BFBFBF28BE9D2B01C0C496684001F309407AC8ECC07E2FA640112069C0C2DB5CC03E0CEBC0544409C0E3F48DC004B228C0E7196E40113B9240A3B7BFC09631AEC09CBDC13FA6132C3FF34BE140EC5A7840C89349BFC4CC0840488FE2BFBAE3C73D796A9A3E58A1AAC00AEA0EC09A3E06C001F753400E1B3740792D13C02AE1EE3E2D2BCD3F3117A63FC85A2E3F569217403A1FD7405406C53FE25CE8BFD99D803FA952603FD4A4C1C02FA9A3BF0C344BBD9BB18A40CFD2F43E1A323E40E779953F8DD992404069EABDADE9703E6B9B3F404DC102C050FC393FFDBC5F40BCB45AC0B69823C0A4258EBF8B0B43C0CAA61BBE"> : tensor<20x30xf32>
    %cst_0 = stablehlo.constant dense<"0x52FF7F3F52FF7F3F000000000000000052FF7F3F00000000000000000000000038B0303D0000000052FF7F3F52FF7F3F000000000000000000000000000000000000000052FF7F3F0000000052FF7F3F000000000000000052FF7F3F0000000052FF7F3F52FF7F3F00000000000000000000000052FF7F3F0000000037AB2B3F52FF7F3F52FF7F3F00000000000000000000000052FF7F3F52FF7F3F00000000000000002980003C0000000000000000000000000000000052FF7F3F00000000000000000000000052FF7F3F000000000000000000000000000000000000000052FF7F3F52FF7F3F0000000052FF7F3F000000000000000052FF7F3F52FF7F3F52FF7F3F00000000000000000000000052FF7F3F52FF7F3F00000000000000000000000052FF7F3F000000000000000052FF7F3F52FF7F3F48E0E03D000000000000000052FF7F3F0000000000000000000000000000000052FF7F3F00000000000000004FF8783E52FF7F3F2E90103E00000000000000000000000052FF7F3F52FF7F3F3BB8B83E0000000052FF7F3F52FF7F3F52FF7F3F52FF7F3F0000000052FF7F3F52FF7F3F52FF7F3F4DF0703D43D0D03D52FF7F3F00000000000000000000000052FF7F3F52FF7F3F0000000033A0A03D52FF7F3F52FF7F3F35A4A43E3BB7373F52FF7F3F0000000052FF7F3F52FF7F3F52FF7F3F52FF7F3F52FF7F3F0000000052FF7F3F52FF7F3F0000000052FF7F3F48E0E03D00000000000000000000000052FF7F3F52FF7F3F52FF7F3F43D0503D0000000052FF7F3F0000000052FF7F3F52FF7F3F0000000052FF7F3F52FF7F3F0000000052FF7F3F0000000052FF7F3F00000000000000000000000052FF7F3F000000002D8C0C3E000000000000000052FF7F3F52FF7F3F000000000000000052FF7F3F52FF7F3F0000000052FF7F3F43D1513F0000000052FF7F3F000000000000000052FF7F3F0000000052FF7F3F4DEF6F3F0000000052FF7F3F52FF7F3F00000000000000000000000052FF7F3F52FF7F3F0000000052FF7F3F0000000052FF7F3F000000000000000052FF7F3F52FF7F3F52FF7F3F52FF7F3F0000000052FF7F3F52FF7F3F52FF7F3F52FF7F3F00000000000000004BEAEA3E0000000052FF7F3F52FF7F3F52FF7F3F37AC2C3E52FF7F3F0000000000000000000000000000000000000000000000000000000052FF7F3F52FF7F3F52FF7F3F0000000052FF7F3F52FF7F3F000000003198183E000000000000000052FF7F3F0000000052FF7F3F000000003BB8B83D52FF7F3F0000000033A0203F0000000052FF7F3F52FF7F3F52FF7F3F000000000000000052FF7F3F52FF7F3F52FF7F3F52FF7F3F52FF7F3F0000000000000000000000000000000000000000000000000000000052FF7F3F52FF7F3F0000000052FF7F3F000000000000000052FF7F3F38B0B03D00000000339E9E3E52FF7F3F36A8283E00000000000000000000000052FF7F3F000000002980803C000000000000000052FF7F3F00000000000000003CBC3C3E000000003FC4C43E00000000000000000000000052FF7F3F52FF7F3F0000000052FF7F3F52FF7F3F0000000052FF7F3F0000000052FF7F3F52FF7F3F319A9A3E000000000000000052FF7F3F0000000052FF7F3F0000000052FF7F3F34A2223F000000000000000052FF7F3F0000000052FF7F3F0000000052FF7F3F52FF7F3F0000000052FF7F3F00000000339E1E3F52FF7F3F0000000052FF7F3F000000000000000052FF7F3F00000000000000002980803B52FF7F3F000000000000000052FF7F3F52FF7F3F000000000000000052FF7F3F52FF7F3F51FCFC3E52FF7F3F52FF7F3F00000000000000000000000052FF7F3F52FF7F3F52FF7F3F000000000000000052FF7F3F0000000000000000000000000000000052FF7F3F52FF7F3F0000000052FF7F3F000000002C89093F52FF7F3F2980003C2980003F52FF7F3F000000002C88883D2F94943E2980803C52FF7F3F2D8E8E3E000000000000000052FF7F3F52FF7F3F0000000052FF7F3F0000000052FF7F3F0000000052FF7F3F52FF7F3F0000000052FF7F3F48E0E03C0000000052FF7F3F0000000052FF7F3F0000000052FF7F3F000000000000000052FF7F3F52FF7F3F52FF7F3F52FF7F3F52FF7F3F0000000052FF7F3F000000000000000052FF7F3F00000000000000000000000052FF7F3F43D0503D0000000052FF7F3F0000000052FF7F3F51FD7D3F00000000000000000000000052FF7F3F0000000052FF7F3F52FF7F3F52FF7F3F4DF0F03D52FF7F3F0000000052FF7F3F00000000000000000000000052FF7F3F00000000000000000000000052FF7F3F000000000000000000000000000000000000000052FF7F3F000000002A84043E00000000000000000000000052FF7F3F52FF7F3F52FF7F3F2980003C52FF7F3F2D8E8E3E000000002C88883E52FF7F3F52FF7F3F52FF7F3F000000000000000052FF7F3F52FF7F3F52FF7F3F52FF7F3F0000000052FF7F3F52FF7F3F52FF7F3F0000000052FF7F3F000000000000000033A0A03D000000002A82823E000000004DF0F03D52FF7F3F2F94143E000000000000000052FF7F3F46DC5C3F000000000000000052FF7F3F000000000000000052FF7F3F3198183E52FF7F3F52FF7F3F4DF0F03D52FF7F3F52FF7F3F52FF7F3F0000000000000000000000000000000052FF7F3F000000000000000052FF7F3F000000000000000000000000000000004AE8683F0000000052FF7F3F0000000052FF7F3F52FF7F3F0000000048E0603D52FF7F3F52FF7F3F0000000052FF7F3F52FF7F3F000000000000000052FF7F3F0000000052FF7F3F0000000052FF7F3F00000000000000000000000045D8D83E000000003BB8B83D52FF7F3F0000000052FF7F3F52FF7F3F52FF7F3F52FF7F3F52FF7F3F000000000000000036AAAA3E0000000052FF7F3F2980803C51FCFC3E0000000052FF7F3F00000000000000000000000052FF7F3F52FF7F3F0000000052FF7F3F00000000000000000000000000000000000000000000000052FF7F3F52FF7F3F000000000000000052FF7F3F4AE6E63E52FF7F3F52FF7F3F0000000052FF7F3F000000002980003C3BB8B83D00000000000000000000000052FF7F3F52FF7F3F0000000048E0603E52FF7F3F52FF7F3F4CEEEE3E52FF7F3F52FF7F3F52FF7F3F0000000052FF7F3F3EC3433F00000000000000000000000052FF7F3F4AE8683E52FF7F3F52FF7F3F52FF7F3F0000000048E0603D52FF7F3F000000002B86063F52FF7F3F0000000000000000000000000000000000000000"> : tensor<20x30xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<20x30xf32>) -> tensor<20x30x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %1 = stablehlo.multiply %0, %0 : (tensor<20x30x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>, tensor<20x30x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>) -> tensor<20x30x!quant.uniform<i8:f32, 0.00392152795604631:-128>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<20x30x!quant.uniform<i8:f32, 0.00392152795604631:-128>>) -> tensor<20x30xf32>
    %3 = stablehlo.custom_call @check.eq(%cst_0, %2) : (tensor<20x30xf32>, tensor<20x30xf32>) -> tensor<i1>
    return %2 : tensor<20x30xf32>
  }
}
